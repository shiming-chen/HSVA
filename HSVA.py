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
import final_classifier_v1 as classifier
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

        if self.manualSeed is None:
            self.manualSeed = random.randint(1, 10000)
        print("Random Seed: ", self.manualSeed)
        random.seed(self.manualSeed)
        torch.manual_seed(self.manualSeed)
        
        feature_dimensions = [2048, self.dataset.aux_data.size(1)]
        discriminator_dim = [1, 1]
        # Here, the encoders and decoders for all modalities are created and put into dict

        self.encoder = {}
        for datatype, dim in zip(self.all_data_sources,feature_dimensions):

            self.encoder[datatype] = models.encoder_template(dim,self.coarse_latent_size,self.device)

            print(str(datatype) + ' ' + str(dim))

        self.decoder = {}
        for datatype, dim in zip(self.all_data_sources,feature_dimensions):
            self.decoder[datatype] = models.decoder_template(self.latent_size,dim,self.device)
        
        self.discriminator = {}
        for datatype, dim in zip(self.all_data_sources,feature_dimensions):
            self.discriminator[datatype] = models.discriminator_template(dim, 1, self.hidden_size_rule[datatype], self.device)
        
        
        seen_class= (self.num_classes-self.num_unseen_classes)
        cls_dim = self.coarse_latent_size + self.dataset.aux_data.size(1)
        self.encoder_z = models.encoder_z(self.coarse_latent_size,self.latent_size, self.device)
        self.decoder_z = models.decoder_z(self.latent_size, 512,self.device)
        
        self.domain_classifier = models.domain_classifier(self.latent_size, self.device)
        self.domain_discriminator = models.domain_discriminator(self.coarse_latent_size, 1, 2048, self.device)
        
        
        self.cls_img = models.class_cls(self.coarse_latent_size, seen_class)
        self.cls_att = models.class_cls(self.coarse_latent_size, seen_class)
        self.cls_z = models.class_cls(self.coarse_latent_size, seen_class)
        
        
        
        print(self.encoder['resnet_features'],self.encoder[self.auxiliary_data_source])
        print(self.encoder_z)
        print(self.decoder['resnet_features'],self.decoder[self.auxiliary_data_source])
        
        # self.domain_label = torch.cuda.LongTensor(self.batch_size)
        # self.domain_label = self.domain_label.cuda()
        self.one = torch.tensor(1, dtype=torch.float)
        self.mone = self.one * -1
        self.one = self.one.cuda()
        self.mone = self.mone.cuda()

        
        
        enc_params = list(self.encoder['resnet_features'].parameters()) + list(self.encoder[self.auxiliary_data_source].parameters())
        dec_params = list(self.decoder['resnet_features'].parameters()) + list(self.decoder[self.auxiliary_data_source].parameters())
        dis_params = list(self.discriminator['resnet_features'].parameters()) + list(self.discriminator[self.auxiliary_data_source].parameters())

        
        self.cls1_opt = optim.Adam(self.cls_img.parameters(), lr=0.001, betas=(0.9, 0.999))
        self.cls2_opt = optim.Adam(self.cls_att.parameters(), lr=0.001, betas=(0.9, 0.999))
        self.cls3_opt = optim.Adam(self.cls_z.parameters(), lr=0.001, betas=(0.9, 0.999))
        
        self.encoder_opt = optim.Adam(self.encoder_z.parameters(),
                                      lr=hyperparameters['lr_gen_model'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
        self.decoder_opt = optim.Adam(self.decoder_z.parameters(),
                                      lr=hyperparameters['lr_gen_model'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
        self.enc_opt  = optim.Adam([p for p in enc_params if p.requires_grad],
                                   lr=hyperparameters['lr_gen_model'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
        self.dec_opt  = optim.Adam([p for p in dec_params if p.requires_grad],
                                   lr=hyperparameters['lr_gen_model'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
        self.dis_opt  = optim.Adam([p for p in dis_params if p.requires_grad],
                                    lr=hyperparameters['lr_gen_model'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
        self.optimizer_domain = optim.Adam(self.domain_classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
        
        # Labels for Adversarial Training
        self.img_label = 0
        self.att_label = 1
        
        self.optimizer_domain_discriminator = optim.Adam(self.domain_discriminator.parameters(),lr=hyperparameters['lr_gen_model'], betas=(0.9, 0.999),
                                                         eps=1e-08, weight_decay=0, amsgrad=True)
        self.optimizer_cls_img = optim.Adam(self.cls_img.parameters(), lr=0.001, weight_decay=0.0005)
        self.optimizer_cls_att = optim.Adam(self.cls_att.parameters(), lr=0.001, weight_decay=0.0005)
        self.optimizer_cls_z = optim.Adam(self.cls_z.parameters(), lr=0.001, weight_decay=0.0005)
        
        # self.cls_criterion = nn.NLLLoss(reduction='none')
        self.cls_criterion = nn.CrossEntropyLoss()
        self.criterion_domain_cls = nn.BCELoss(reduction='none')
        if self.reco_loss_function=='l2':
            self.reconstruction_criterion = nn.MSELoss(size_average=False)

        elif self.reco_loss_function=='l1':
            self.reconstruction_criterion = nn.L1Loss(size_average=False)
        
    def reparameterize(self, mu, logvar):
        if self.reparameterize_with_noise:
            sigma = torch.exp(0.1 * logvar)
            eps = torch.cuda.FloatTensor(logvar.size()[0],1).normal_(0,1)
            eps  = eps.expand(sigma.size())
            return mu + sigma*eps
        else:
            return mu

    def forward(self):
        pass

    def map_label(self,label, classes):
        mapped_label = torch.LongTensor(label.size()).to(self.device)
        for i in range(classes.size(0)):
            mapped_label[label==classes[i]] = i

        return mapped_label
        
    def calc_gradient_penalty(self, netD,real_data, fake_data):
        alpha = torch.rand(self.batch_size, 1)
        alpha = alpha.expand(real_data.size())
        if self.device:
            alpha = alpha.cuda()
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        if self.device:
            interpolates = interpolates.cuda()
        interpolates = Variable(interpolates, requires_grad=True)
        disc_interpolates = netD(interpolates)
        ones = torch.ones(disc_interpolates.size())
        if self.device:
            ones = ones.cuda()
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=ones,
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
        return gradient_penalty
        
    def reset_grad(self):
        # self.decoder_opt.zero_grad()
        self.encoder_opt.zero_grad()
        self.enc_opt.zero_grad()
        self.dec_opt.zero_grad()
        self.cls1_opt.zero_grad()
        self.cls2_opt.zero_grad()
        # self.cls3_opt.zero_grad()
    
    def discrepancy(self, out1, out2):
        return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))
        
        
    def discrepancy_slice_wasserstein(self, p1, p2):
        p1 = F.softmax(p1)
        p2 = F.softmax(p2)
        s = p1.shape
        if s[1]>1:
            proj = torch.randn(s[1], 128).cuda()
            proj *= torch.rsqrt(torch.sum(torch.mul(proj, proj), 0, keepdim=True))
            p1 = torch.matmul(p1, proj)
            p2 = torch.matmul(p2, proj)
        p1 = torch.topk(p1, s[0], dim=0)[0]
        p2 = torch.topk(p2, s[0], dim=0)[0]
        dist = p1-p2
        wdist = torch.mean(torch.mul(dist, dist))
        
        return wdist
    
    def get_CORAL_loss(self, source, target):
        batch_size = source.data.shape[0]

        # source covariance
        xm = torch.mean(source, 0, keepdim=True) - source
        xc = xm.t() @ xm / (batch_size - 1)

        # target covariance
        xmt = torch.mean(target, 0, keepdim=True) - target
        xct = xmt.t() @ xmt / (batch_size - 1)

        # frobenius norm between source and target
        loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
        return loss
    
    
    
    def loss_fn(recon_x, x, mean, log_var):
        BCE = torch.nn.functional.binary_cross_entropy(recon_x+1e-12, x.detach(),reduction='sum')
        BCE = BCE.sum()/ x.size(0)
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())/ x.size(0)
        #return (KLD)
        return (BCE + KLD)
               

    def WeightedL1(pred, gt):
        wt = (pred-gt).pow(2)
        wt /= wt.sum(1).sqrt().unsqueeze(1).expand(wt.size(0),wt.size(1))
        loss = wt * (pred-gt).abs()
        return loss.sum()/loss.size(0)
        
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
            unseen_coarse_features = self.encoder['resnet_features'](unseen_test_feat)
            mu1, var1 = self.encoder_z(unseen_coarse_features)
            # test_unseen_X = unseen_coarse_features.to(self.device).data
            test_unseen_X = self.reparameterize(mu1, var1).to(self.device).data
            test_unseen_Y = test_unseen_label.to(self.device)

            seen_coarse_features = self.encoder['resnet_features'](seen_test_feat)
            mu2, var2 = self.encoder_z(seen_coarse_features)
            # test_seen_X = seen_coarse_features.to(self.device).data  # coarse_feature
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
                    # z = coarse_commom  ## coarse_feature
                    z = self.reparameterize(mu_, logvar_)  ## fine_feature
                    return z
                else:
                    return torch.cuda.FloatTensor([])

            z_seen_img   = convert_datapoints_to_z(img_seen_feat, self.encoder['resnet_features'])
            z_unseen_img = convert_datapoints_to_z(img_unseen_feat, self.encoder['resnet_features'])
            z_seen_att = convert_datapoints_to_z(att_seen_feat, self.encoder[self.auxiliary_data_source])
            z_unseen_att = convert_datapoints_to_z(att_unseen_feat, self.encoder[self.auxiliary_data_source])
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
                    if best_gzsl_acc < cls.H:
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

    
       
    def gen_update(self, img, att, label, unseen_att):
        f1 = 1.0*(self.current_epoch - self.warmup['cross_reconstruction']['start_epoch'] )/(1.0*( self.warmup['cross_reconstruction']['end_epoch']- self.warmup['cross_reconstruction']['start_epoch']))
        f1 = f1*(1.0*self.warmup['cross_reconstruction']['factor'])
        reconstruction_factor_cross = torch.cuda.FloatTensor([min(max(f1,0),self.warmup['cross_reconstruction']['factor'])])

        f2 = 1.0 * (self.current_epoch - self.warmup['beta']['start_epoch']) / ( 1.0 * (self.warmup['beta']['end_epoch'] - self.warmup['beta']['start_epoch']))
        f2 = f2 * (1.0 * self.warmup['beta']['factor'])
        beta = torch.cuda.FloatTensor([min(max(f2, 0), self.warmup['beta']['factor'])])

        f3 = 1.0*(self.current_epoch - self.warmup['distance']['start_epoch'] )/(1.0*( self.warmup['distance']['end_epoch']- self.warmup['distance']['start_epoch']))
        f3 = f3*(1.0*self.warmup['distance']['factor'])
        distance_factor = torch.cuda.FloatTensor([min(max(f3,0),self.warmup['distance']['factor'])]) 
        
       #======================================================================================
       # train first-adaptation
       #======================================================================================
        ### train classifiers
        self.reset_grad()
        coarse_common_img= self.encoder['resnet_features'](img)
        coarse_common_att = self.encoder[self.auxiliary_data_source](att)
        pre_s1 = self.cls_img(coarse_common_img.cuda(), att.cuda())
        pre_s2 = self.cls_att(coarse_common_img.cuda(), att.cuda())
        pre_t1 = self.cls_img(coarse_common_att.cuda(), att.cuda())
        pre_t2 = self.cls_att(coarse_common_att.cuda(), att.cuda())
        loss_s1 = self.cls_criterion(pre_s1, label)
        loss_s2 = self.cls_criterion(pre_s2, label)
        loss_t1 = self.cls_criterion(pre_t1, label)
        loss_t2 = self.cls_criterion(pre_t2, label)
        loss_s = loss_s1 + loss_s2
        loss_t = loss_t1 + loss_t2
        loss_cls = loss_s + loss_t
        loss_cls.backward()
        self.cls1_opt.step()
        self.cls2_opt.step()
        self.enc_opt.step()
        self.reset_grad()
        
        
        ### Maximize the discrepancy 
        coarse_common_img= self.encoder['resnet_features'](img)
        coarse_common_att = self.encoder[self.auxiliary_data_source](att)
        pre_s1 = self.cls_img(coarse_common_img.cuda(), att.cuda())
        pre_s2 = self.cls_att(coarse_common_img.cuda(), att.cuda())
        pre_t1 = self.cls_img(coarse_common_att.cuda(), att.cuda())
        pre_t2 = self.cls_att(coarse_common_att.cuda(), att.cuda())
        loss_s1 = self.cls_criterion(pre_s1, label)
        loss_s2 = self.cls_criterion(pre_s2, label)
        loss_t1 = self.cls_criterion(pre_t1, label)
        loss_t2 = self.cls_criterion(pre_t2, label)
        loss_s = loss_s1 + loss_s2
        loss_t = loss_t1 + loss_t2
        loss_cls = loss_s + loss_t
        if self.adapt_mode == 'MCD':
            loss_dis_s = self.discrepancy(pre_s1, pre_s2)
            loss_dis_t = self.discrepancy(pre_t1, pre_t2)
            loss_dis = self.discrepancy(pre_s1, pre_t1)
        else:
            loss_dis_s = self.discrepancy_slice_wasserstein(pre_s1, pre_s2)
            loss_dis_t = self.discrepancy_slice_wasserstein(pre_t1, pre_t2)
            loss_dis = self.discrepancy_slice_wasserstein(pre_s1, pre_t1)
        loss = loss_s + loss_t - distance_factor * (loss_dis_s + loss_dis_t)# + loss_dis)
        loss.backward()
        self.cls1_opt.step()
        self.cls2_opt.step()
        self.reset_grad()
        
        #### Minimize the discrepancy 
        for i in range(2):
            coarse_common_img= self.encoder['resnet_features'](img)
            coarse_common_att = self.encoder[self.auxiliary_data_source](att)
            pre_s1 = self.cls_img(coarse_common_img.cuda(), att.cuda())
            pre_s2 = self.cls_att(coarse_common_img.cuda(), att.cuda())
            pre_t1 = self.cls_img(coarse_common_att.cuda(), att.cuda())
            pre_t2 = self.cls_att(coarse_common_att.cuda(), att.cuda())
            loss_s1 = self.cls_criterion(pre_s1, label)
            loss_s2 = self.cls_criterion(pre_s2, label)
            loss_t1 = self.cls_criterion(pre_t1, label)
            loss_t2 = self.cls_criterion(pre_t2, label)
            loss_s = loss_s1 + loss_s2
            loss_t = loss_t1 + loss_t2
            loss_cls = loss_s + loss_t
            if self.adapt_mode == 'MCD':
                loss_dis_s = self.discrepancy(pre_s1, pre_s2)
                loss_dis_t = self.discrepancy(pre_t1, pre_t2)
                loss_dis = self.discrepancy(pre_s1, pre_t1)
            else:
                loss_dis_s = self.discrepancy_slice_wasserstein(pre_s1, pre_s2)
                loss_dis_t = self.discrepancy_slice_wasserstein(pre_t1, pre_t2)
                loss_dis = self.discrepancy_slice_wasserstein(pre_s1, pre_t1)
            loss = distance_factor * (loss_dis_s + loss_dis_t)# + loss_dis)
            loss.backward()
            self.enc_opt.step()
            self.reset_grad()
        




        
        #======================================================================================
        # train second-adaptation
        #======================================================================================
        
        
         # encode
        coarse_common_img = self.encoder['resnet_features'](img)
        mu_img, logvar_img = self.encoder_z(coarse_common_img)
        z_from_img = self.reparameterize(mu_img, logvar_img)
        
        coarse_common_att = self.encoder[self.auxiliary_data_source](att)
        mu_att, logvar_att = self.encoder_z(coarse_common_att)
        z_from_att = self.reparameterize(mu_att, logvar_att)
         
        # encode unseen_att
        coarse_common_unseen_att = self.encoder[self.auxiliary_data_source](unseen_att)
        mu_unseen_att, logvar_unseen_att = self.encoder_z(coarse_common_unseen_att)
        z_from_unseen_att = self.reparameterize(mu_unseen_att, logvar_unseen_att)
         
        # decode (within domain)
        img_from_img = self.decoder['resnet_features'](z_from_img)
        att_from_att = self.decoder[self.auxiliary_data_source](z_from_att)
        att_from_unseen_att = self.decoder[self.auxiliary_data_source](z_from_unseen_att)
        
        
        
        # decode (cross domain)
        img_from_att = self.decoder['resnet_features'](z_from_att)
        att_from_img = self.decoder[self.auxiliary_data_source](z_from_img)
        
        # reconstruction loss
        self.reconstruction_loss_within = self.reconstruction_criterion(img_from_img, img) + \
                                     self.reconstruction_criterion(att_from_att, att)
        self.reconstruction_loss_cross = self.reconstruction_criterion(img_from_att, img) + \
                                     self.reconstruction_criterion(att_from_img, att)
        
        self.KLD_within = (0.5 * torch.sum(1 + logvar_att - mu_att.pow(2) - logvar_att.exp())) + \
              (0.5 * torch.sum(1 + logvar_img - mu_img.pow(2) - logvar_img.exp()))
        
        
        
        self.reset_grad()
        
        # Distribution Alignment
        self.distance_seen_seen = torch.sqrt(torch.sum((mu_img - mu_att) ** 2, dim=1) + \
                              torch.sum((torch.sqrt(logvar_img.exp()) - torch.sqrt(logvar_att.exp())) ** 2, dim=1)).sum()
        
        
        self.distance_seen_unseen_att = torch.sqrt(torch.sum((mu_att - mu_unseen_att) ** 2, dim=1) + \
                        torch.sum((torch.sqrt(logvar_att.exp()) - torch.sqrt(logvar_unseen_att.exp())) ** 2, dim=1)).sum()
        
        self.distance_seen_unseen_img = torch.sqrt(torch.sum((mu_img - mu_unseen_att) ** 2, dim=1) + \
                        torch.sum((torch.sqrt(logvar_img.exp()) - torch.sqrt(logvar_unseen_att.exp())) ** 2, dim=1)).sum()
        
        self.distance = torch.max(5 + self.distance_seen_seen - self.distance_seen_unseen_att,torch.tensor(0.0).cuda())
        
        coral_seen_unseen = self.get_CORAL_loss(z_from_img, z_from_unseen_att)

        
        self.loss_g = (self.reconstruction_loss_within - beta*self.KLD_within)
        if self.reconstruction_loss_cross>0:
            self.loss_g += reconstruction_factor_cross * self.reconstruction_loss_cross
        if distance_factor >0:
           self.loss_g += distance_factor * (self.distance_seen_seen - coral_seen_unseen)
        self.loss_g.backward()
        self.dec_opt.step()
        self.encoder_opt.step()
        self.enc_opt.step()
        self.reset_grad()
        
        
        
        
        return self.distance_seen_seen, self.loss_g
    

    def train_vae(self):

        losses = []


        self.dataset.unseenclasses =self.dataset.unseenclasses.long().cuda()
        self.dataset.seenclasses =self.dataset.seenclasses.long().cuda()
        #leave both statements
        self.train()
        self.reparameterize_with_noise = True

        print('train for reconstruction')
        
        best_H = 0
        best_acc = 0
        best_unseen = 0
        for epoch in range(0, self.nepoch ):
            self.current_epoch = epoch

            i=-1
            for iters in range(0, self.dataset.ntrain, self.batch_size):
                i+=1

                data_from_modalities = self.dataset.next_batch(self.batch_size)
                for j in range(len(data_from_modalities)):
                    data_from_modalities[j] = data_from_modalities[j].to(self.device)
                    data_from_modalities[j].requires_grad = False
                
                seen_label = self.map_label(data_from_modalities[2], self.dataset.seenclasses)
                
                
                data_from_unseen = self.dataset.next_unseen_batch(self.batch_size) 
                for j in range(len(data_from_unseen)):
                    data_from_unseen[j] = data_from_unseen[j].to(self.device)
                    data_from_unseen[j].requires_grad = False
                # unseen_label = seen_label = label = self.map_label(data_from_unseen[1], self.dataset.unseenclasses)

                distance,loss_gen = self.gen_update(data_from_modalities[0], data_from_modalities[1], seen_label, data_from_unseen[0])
                if i%50==0:

                    # print('epoch ' + str(epoch) + ' | iter ' + str(i) + '\t'+
                    # ' | loss_dis ' +  str(loss_dis)[7:15] + ' | loss_gen ' +  str(loss_gen)[8:15])
                    print('epoch ' + str(epoch) + ' | iter ' + str(i) + '\t'+ ' | loss_gen ' +  str(loss_gen)[8:14] + ' | distance ' +  str(distance)[7:13])


            

            # turn into evaluation mode:
            for key, value in self.encoder.items():
                self.encoder[key].eval()
            for key, value in self.decoder.items():
                self.decoder[key].eval()
                
            if epoch>=100:
                if self.generalized:
                    unseen, seen, H = self.train_classifier(current_epoch=epoch)
                    if best_H<H:# and best_unseen< unseen:
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
    
    
    
 
