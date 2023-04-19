import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.bias.data.fill_(0)

        nn.init.xavier_uniform_(m.weight,gain=0.5)


    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


        
class encoder_z(nn.Module):

    def __init__(self,input_dim, latent_size,device):
        super(encoder_z_v5,self).__init__()
        self.layer_sizes = [input_dim, 2048, latent_size]
        modules = []
        for i in range(len(self.layer_sizes)-2):
            modules.append(nn.Linear(self.layer_sizes[i],self.layer_sizes[i+1]))
            modules.append(nn.LeakyReLU(0.2, True))
        self.feature_encoder = nn.Sequential(*modules)
        self._mu = nn.Linear(in_features=self.layer_sizes[-2], out_features=latent_size)
        self._logvar = nn.Linear(in_features=self.layer_sizes[-2], out_features=latent_size)
        self.apply(weights_init)
        self.to(device)


    def forward(self,x):
        z = self.feature_encoder(x)
        mu =  self._mu(z)
        logvar = self._logvar(z)

        return mu, logvar




        
class encoder_template(nn.Module):

    def __init__(self,input_dim,output_dim,device):
        super(encoder_template_v5,self).__init__()
        self.feature_encoder = nn.Sequential(nn.Linear(input_dim,4096),nn.LeakyReLU(0.2, True),nn.Linear(4096,output_dim))
        self.apply(weights_init)
        self.to(device)


    def forward(self,x):
        h = self.feature_encoder(x)
        return h



class decoder_template(nn.Module):

    def __init__(self,input_dim,output_dim,device):
        super(decoder_template,self).__init__()
        self.layer_sizes = [input_dim, 4096 , output_dim]
        self.feature_decoder = nn.Sequential(nn.Linear(input_dim,self.layer_sizes[1]),nn.LeakyReLU(0.2, True),nn.Linear(self.layer_sizes[1],output_dim))
        self.apply(weights_init)
        self.to(device)
    def forward(self,x):

        return self.feature_decoder(x)




class decoder_z(nn.Module):

    def __init__(self,latent_size, output_dim, device):
        super(decoder_z,self).__init__()


        self.layer_sizes = [latent_size, 2048, output_dim]

        self.feature_decoder = nn.Sequential(nn.Linear(latent_size,self.layer_sizes[1]),nn.LeakyReLU(0.2, True),nn.Linear(self.layer_sizes[1],output_dim))

        self.apply(weights_init)

        self.to(device)
    def forward(self,x):

        return self.feature_decoder(x)


        
class domain_classifier(nn.Module):
    def __init__(self, hidden_size,device):
        super(domain_classifier, self).__init__()
        self.classify = nn.Sequential(nn.Linear(hidden_size,512),nn.LeakyReLU(0.2, True),nn.Linear(512,1),nn.Sigmoid())
        self.to(device)
    def forward(self, x): 
        output = self.classify(x)
        output = output.view(-1)
        return output  

    
class domain_discriminator(nn.Module):
    def __init__(self,input_dim,output_dim,hiden_size,device): 
        super(domain_discriminator, self).__init__()
        

        self.discriminator = nn.Sequential(nn.Linear(input_dim,output_dim),nn.LeakyReLU(0.2, True),nn.Linear(output_dim,output_dim))

        self.apply(weights_init)

        self.to(device)
        
    def forward(self,x):
        return self.discriminator(x)
    

class class_cls(nn.Module):
    def __init__(self, input_dim, nclass):
        super(class_cls, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1_fc = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, nclass)
        self.bn_fc2 = nn.BatchNorm1d(nclass)
        self.apply(weights_init)
    def forward(self, x, att): 
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = self.fc2(x)
        return x
