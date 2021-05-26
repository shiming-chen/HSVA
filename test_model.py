#vaemodel
import copy
import torch
import sys
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torch.utils import data
from data_loader import DATA_LOADER as dataloader
import final_classifier as  classifier
import models
import random
import os

class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim,nclass)
        self.logic = nn.LogSoftmax(dim=1)
        self.lossfunction =  nn.NLLLoss()

    def forward(self, x):
        o = self.logic(self.fc(x))
        return o

class Model(nn.Module):

    def __init__(self,hyperparameters):
        super(Model,self).__init__()
        self.selected_cls = hyperparameters['classifier']
        self.device = hyperparameters['device']
        self.auxiliary_data_source = hyperparameters['auxiliary_data_source']
        self.all_data_sources  = ['resnet_features',self.auxiliary_data_source]
        self.DATASET = hyperparameters['dataset']
        self.num_shots = hyperparameters['num_shots']
        self.latent_size = hyperparameters['latent_size']
        self.batch_size = hyperparameters['batch_size']
        self.hidden_size_rule = hyperparameters['hidden_size_rule']
        self.warmup = hyperparameters['model_specifics']['warmup']
        self.generalized = hyperparameters['generalized']
        self.classifier_batch_size = 32
        self.img_seen_samples   = hyperparameters['samples_per_class'][self.DATASET][0]
        self.att_seen_samples   = hyperparameters['samples_per_class'][self.DATASET][1]
        self.att_unseen_samples = hyperparameters['samples_per_class'][self.DATASET][2]
        self.img_unseen_samples = hyperparameters['samples_per_class'][self.DATASET][3]
        self.reco_loss_function = hyperparameters['loss']
        self.nepoch = hyperparameters['epochs']
        self.lr_cls = hyperparameters['lr_cls']
        self.result_root = hyperparameters['result_root']
        self.coarse_latent_size = hyperparameters['coarse_latent_size']
        self.adapt_mode = hyperparameters['adapt_mode']
        self.cross_reconstruction = hyperparameters['model_specifics']['cross_reconstruction']
        self.cls_train_epochs = hyperparameters['cls_train_steps']
        self.dataset = dataloader( self.DATASET, copy.deepcopy(self.auxiliary_data_source) , device= self.device )
        
        if not os.path.exists(os.path.join(self.result_root, self.DATASET)):
            os.makedirs(os.path.join(self.result_root, self.DATASET))
        
        
        if self.DATASET=='CUB':
            self.num_classes=200
            self.num_unseen_classes = 50
            self.manualSeed = 3483
        elif self.DATASET=='SUN':
            self.num_classes=717
            self.num_unseen_classes = 72
            self.manualSeed = 4115
        elif self.DATASET=='AWA1' or self.DATASET=='AWA2':
            self.num_classes=50
            self.num_unseen_classes = 10
            self.manualSeed = 4115
        elif self.DATASET=='APY':
            self.num_classes=32
            self.num_unseen_classes = 12
            self.manualSeed = 9182
        elif self.DATASET=='FLO':
            self.num_classes=102
            self.num_unseen_classes = 20
            self.manualSeed = 806

        feature_dim = [2048, self.dataset.aux_data.size(1)]
        self.encoder_v = models.encoder_template_v5(feature_dim[0],self.coarse_latent_size, self.device)
        self.encoder_a = models.encoder_template_v5(feature_dim[1],self.coarse_latent_size,self.device)
        self.encoder_z = models.encoder_z_v5(self.coarse_latent_size,self.latent_size, self.device)
        checkpoint_encv = torch.load(os.path.join(self.result_root, self.DATASET, 'checkpoint_encv.pth.tar'), map_location='cpu')['encoder_v']
        checkpoint_enca = torch.load(os.path.join(self.result_root, self.DATASET, 'checkpoint_enca.pth.tar'), map_location='cpu')['encoder_a']
        checkpoint_encz = torch.load(os.path.join(self.result_root, self.DATASET, 'checkpoint_encz.pth.tar'), map_location='cpu')['encoder_z']
        self.encoder_v.load_state_dict(checkpoint_encv)
        self.encoder_a.load_state_dict(checkpoint_enca)
        self.encoder_z.load_state_dict(checkpoint_encz)
        self.encoder_v.eval()
        self.encoder_a.eval()
        self.encoder_z.eval()
        
    def reparameterize(self, mu, logvar):
        if self.reparameterize_with_noise:
            # sigma = torch.exp(logvar)
            # eps = torch.cuda.FloatTensor(logvar.size()[0],1).normal_(0,1)
            # eps  = eps.expand(sigma.size())
            sigma = torch.exp(0.1 * logvar)
            eps = torch.cuda.FloatTensor(logvar.size()[0],1).normal_(0,1)
            eps  = eps.expand(sigma.size())
            return mu + sigma*eps
        else:
            return mu



    def map_label(self,label, classes):
        mapped_label = torch.LongTensor(label.size()).to(self.device)
        for i in range(classes.size(0)):
            mapped_label[label==classes[i]] = i

        return mapped_label
        

    

        
    def train_classifier(self, current_epoch):

        if self.num_shots > 0 :
            print('================  transfer features from test to train ==================')
            self.dataset.transfer_features(self.num_shots, num_queries='num_features')

        history = []  # stores accuracies


        cls_seenclasses = self.dataset.seenclasses
        cls_unseenclasses = self.dataset.unseenclasses


        train_seen_feat = self.dataset.data['train_seen']['resnet_features']
        train_seen_label = self.dataset.data['train_seen']['labels']

        unseenclass_aux_data = self.dataset.unseenclass_aux_data  # access as unseenclass_aux_data['resnet_features'], unseenclass_aux_data['attributes']
        seenclass_aux_data = self.dataset.seenclass_aux_data

        unseen_corresponding_labels = self.dataset.unseenclasses.long().to(self.device)
        seen_corresponding_labels = self.dataset.seenclasses.long().to(self.device)


        # The resnet_features for testing the classifier are loaded here
        unseen_test_feat = self.dataset.data['test_unseen'][
            'resnet_features']  # self.dataset.test_unseen_feature.to(self.device)
        seen_test_feat = self.dataset.data['test_seen'][
            'resnet_features']  # self.dataset.test_seen_feature.to(self.device)
        test_seen_label = self.dataset.data['test_seen']['labels']  # self.dataset.test_seen_label.to(self.device)
        test_unseen_label = self.dataset.data['test_unseen']['labels']  # self.dataset.test_unseen_label.to(self.device)

        train_unseen_feat = self.dataset.data['train_unseen']['resnet_features']
        train_unseen_label = self.dataset.data['train_unseen']['labels']


        # in ZSL mode:
        if self.generalized == False:
            # there are only 50 classes in ZSL (for CUB)
            # unseen_corresponding_labels =list of all unseen classes (as tensor)
            # test_unseen_label = mapped to 0-49 in classifier function
            # those are used as targets, they have to be mapped to 0-49 right here:

            unseen_corresponding_labels = self.map_label(unseen_corresponding_labels, unseen_corresponding_labels)

            if self.num_shots > 0:
                # not generalized and at least 1 shot means normal FSL setting (use only unseen classes)
                train_unseen_label = self.map_label(train_unseen_label, cls_unseenclasses)

            # for FSL, we train_seen contains the unseen class examples
            # for ZSL, train seen label is not used
            # if self.num_shots>0:
            #    train_seen_label = self.map_label(train_seen_label,cls_unseenclasses)

            test_unseen_label = self.map_label(test_unseen_label, cls_unseenclasses)

            # map cls unseenclasses last
            cls_unseenclasses = self.map_label(cls_unseenclasses, cls_unseenclasses)


        if self.generalized:
            clf = LINEAR_LOGSOFTMAX(self.latent_size, self.num_classes)
        else:
            clf = LINEAR_LOGSOFTMAX(self.latent_size, self.num_unseen_classes)


        clf.apply(models.weights_init)

        with torch.no_grad():

            ####################################
            # preparing the test set
            # convert raw test data into z vectors
            ####################################

            self.reparameterize_with_noise = False
            unseen_coarse_features = self.encoder_v(unseen_test_feat)
            mu1, var1 = self.encoder_z(unseen_coarse_features)
            test_unseen_X = self.reparameterize(mu1, var1).to(self.device).data
            test_unseen_Y = test_unseen_label.to(self.device)

            seen_coarse_features = self.encoder_v(seen_test_feat)
            mu2, var2 = self.encoder_z(seen_coarse_features)
            test_seen_X = self.reparameterize(mu2, var2).to(self.device).data   ## fine_feature
            test_seen_Y = test_seen_label.to(self.device)

            ####################################
            # preparing the train set:
            # chose n random image features per
            # class. If n exceeds the number of
            # image features per class, duplicate
            # some. Next, convert them to
            # latent z features.
            ####################################

            self.reparameterize_with_noise = True

            def sample_train_data_on_sample_per_class_basis(features, label, sample_per_class):
                sample_per_class = int(sample_per_class)

                if sample_per_class != 0 and len(label) != 0:

                    classes = label.unique()

                    for i, s in enumerate(classes):

                        features_of_that_class = features[label == s, :]  # order of features and labels must coincide
                        # if number of selected features is smaller than the number of features we want per class:
                        multiplier = torch.ceil(torch.cuda.FloatTensor(
                            [max(1, sample_per_class / features_of_that_class.size(0))])).long().item()

                        features_of_that_class = features_of_that_class.repeat(multiplier, 1)

                        if i == 0:
                            features_to_return = features_of_that_class[:sample_per_class, :]
                            labels_to_return = s.repeat(sample_per_class)
                        else:
                            features_to_return = torch.cat(
                                (features_to_return, features_of_that_class[:sample_per_class, :]), dim=0)
                            labels_to_return = torch.cat((labels_to_return, s.repeat(sample_per_class)),
                                                         dim=0)

                    return features_to_return, labels_to_return
                else:
                    return torch.cuda.FloatTensor([]), torch.cuda.LongTensor([])


            # some of the following might be empty tensors if the specified number of
            # samples is zero :

            img_seen_feat,   img_seen_label   = sample_train_data_on_sample_per_class_basis(
                train_seen_feat,train_seen_label,self.img_seen_samples )

            img_unseen_feat, img_unseen_label = sample_train_data_on_sample_per_class_basis(
                train_unseen_feat, train_unseen_label, self.img_unseen_samples )

            att_unseen_feat, att_unseen_label = sample_train_data_on_sample_per_class_basis(
                    unseenclass_aux_data,
                    unseen_corresponding_labels,self.att_unseen_samples )

            att_seen_feat, att_seen_label = sample_train_data_on_sample_per_class_basis(
                seenclass_aux_data,
                seen_corresponding_labels, self.att_seen_samples)

            def convert_datapoints_to_z(features, encoder):
                if features.size(0) != 0:
                    coarse_commom = encoder(features)
                    mu_, logvar_ = self.encoder_z(coarse_commom)
                    z = self.reparameterize(mu_, logvar_)  ## fine_feature
                    return z
                else:
                    return torch.cuda.FloatTensor([])

            z_seen_img   = convert_datapoints_to_z(img_seen_feat, self.encoder_v)
            z_unseen_img = convert_datapoints_to_z(img_unseen_feat, self.encoder_v)
            z_seen_att = convert_datapoints_to_z(att_seen_feat, self.encoder_a)
            z_unseen_att = convert_datapoints_to_z(att_unseen_feat, self.encoder_a)
            train_Z = [z_seen_img, z_unseen_img, z_seen_att, z_unseen_att]
            train_L = [img_seen_label    , img_unseen_label,att_seen_label,att_unseen_label]
            
            # empty tensors are sorted out
            train_X = [train_Z[i] for i in range(len(train_Z)) if train_Z[i].size(0) != 0]
            train_Y = [train_L[i] for i in range(len(train_L)) if train_Z[i].size(0) != 0]

            train_X = torch.cat(train_X, dim=0)
            train_Y = torch.cat(train_Y, dim=0)

        ############################################################
        ##### initializing the classifier and train one epoch
        ############################################################
        cls = classifier.CLASSIFIER(clf, train_X, train_Y, test_seen_X, test_seen_Y, test_unseen_X,
                                    test_unseen_Y,
                                    cls_seenclasses, cls_unseenclasses,
                                    self.num_classes, self.device, self.selected_cls, self.lr_cls, 0.5, 1,
                                    self.classifier_batch_size,
                                    self.generalized)
        best_gzsl_acc = 0
        best_zsl_acc = 0
        if self.selected_cls == 'softmax':
            for k in range(self.cls_train_epochs):
                if self.generalized:
                    cls.acc_seen, cls.acc_unseen, cls.H = cls.fit()
                    if best_gzsl_acc <= cls.H:
                        best_gzsl_acc = cls.H
                        best_gzsl_epoch = k
                        best_seen, best_unseen, best_H = cls.acc_seen, cls.acc_unseen, cls.H
                else:
                    cls.acc = cls.fit_zsl()
                    if best_zsl_acc < cls.acc:
                        best_zsl_epoch= k
                        best_unseen = cls.acc
            if self.generalized:
                print('[epoch=%.1f] unseen=%.3f, seen=%.3f, h=%.3f  - - - - - - ' % (
                    current_epoch, best_unseen, best_seen, best_H), end="")
                return best_unseen, best_seen, best_H
                
                   
            else:
                print('[epoch=%.1f] acc=%.3f - - - - - - ' % (current_epoch, best_unseen), end="")
                return best_unseen
        else:
            if self.generalized:
                best_seen, best_unseen, best_H = cls.acc_seen, cls.acc_unseen, cls.H
                print('[epoch=%.1f] unseen=%.3f, seen=%.3f, h=%.3f  - - - - - - ' % (
                    current_epoch, best_unseen, best_seen, best_H), end="")
                return best_unseen, best_seen, best_H
            else:
                for k in range(self.cls_train_epochs):
                    cls.acc = cls.fit_zsl()
                    if best_zsl_acc <= cls.acc:
                        best_zsl_epoch= k
                        best_unseen = cls.acc
                print('[epoch=%.1f] acc=%.3f - - - - - - ' % (current_epoch, best_unseen), end="")
                return best_unseen

    
       
    

    def test(self):



        self.dataset.unseenclasses =self.dataset.unseenclasses.long().cuda()
        self.dataset.seenclasses =self.dataset.seenclasses.long().cuda()
        self.reparameterize_with_noise = True

        
        best_H = 0
        best_acc = 0
        best_unseen = 0
        for epoch in range(0, self.nepoch ):
            self.current_epoch = epoch
            if epoch>=0:
                if self.generalized:
                    unseen, seen, H = self.train_classifier(current_epoch=epoch)
                    if best_H<=H:# and best_unseen< unseen:
                        best_gzsl_epoch= epoch
                        best_unseen, best_seen, best_H= unseen, seen, H
                    print('[best_epoch=%.1f] best_unseen=%.3f, best_seen=%.3f, best_h=%.3f' % (
                    best_gzsl_epoch, best_unseen, best_seen, best_H))
                    
                       
                else:
                    # return 0, torch.tensor(cls.acc).item(), 0, history
                    acc = self.train_classifier(current_epoch=epoch)
                    if best_acc<acc:
                        best_epoch = epoch
                        best_acc = acc
                    print('[best_epoch=%.1f] best_acc=%.3f' % (best_epoch, best_acc))
    
    
    
 
