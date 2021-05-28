
### execute this function to train and test the vae-model

from test_model import Model
import numpy as np
import pickle
import torch
import os
import argparse
import time
import warnings
warnings.filterwarnings("ignore")

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()

parser.add_argument('--dataset')
parser.add_argument('--latent_size',type=int)
parser.add_argument('--generalized', type = str2bool)
args = parser.parse_args()


########################################
# the basic hyperparameters
########################################
hyperparameters = {
    'num_shots': 0,
    'device': 'cuda',
    'model_specifics': {'cross_reconstruction': True,
                       'name': 'CADA',
                       'distance': 'wasserstein',
                       'warmup': {'beta': {'factor': 0.25,
                                           'end_epoch': 90,
                                           'start_epoch': 0},
                                  'cross_reconstruction': {'factor': 2.37,
                                                           'end_epoch': 75,
                                                           'start_epoch': 21},
                                  'distance': {'factor': 8.0,
                                               'end_epoch': 25,
                                               'start_epoch': 0}}},

    'lr_gen_model': 0.00015,
    'generalized': True,
    'batch_size': 50,
    'samples_per_class': {'SUN': (200, 0, 400, 0),
                          'APY': (200, 0, 400, 0),
                          'CUB': (200, 0, 400, 0),
                          'AWA2': (200, 0, 400, 0),
                          'FLO': (200, 0, 400, 0),
                          'AWA1': (200, 0, 400, 0)},
    'epochs': 200,
    'loss': 'l1',
    'auxiliary_data_source' : 'attributes',
    'lr_cls': 0.001,
    'dataset': 'CUB',
    'hidden_size_rule': {'resnet_features': (4096, 4096),
                        'attributes': (4096, 4096),
                        'sentences': (4096, 4096) },
    'coarse_latent_size': 2048,
    'latent_size': 64,
    'recon_x_cyc_w': 0.5,
    'adapt_mode': 'SWD',               #MCD or SWD
    'classifier': 'softmax',          #knn or softmax
    'result_root': '/home/shimingchen/ZSL/Feature-Matching/model/result'
}

# The training epochs for the final classifier, for early stopping,
# as determined on the validation spit

cls_train_steps = [
      {'dataset': 'SUN',  'latent_size': 256, 'generalized': True, 'cls_train_steps': 40},
      {'dataset': 'SUN',  'latent_size': 256, 'generalized': False, 'cls_train_steps': 30},
      {'dataset': 'AWA1', 'latent_size': 64, 'generalized': True, 'cls_train_steps': 33},
      {'dataset': 'AWA1', 'latent_size': 64, 'generalized': False, 'cls_train_steps': 25},
      {'dataset': 'CUB',  'latent_size': 64, 'generalized': True, 'cls_train_steps': 23},
      {'dataset': 'CUB',  'latent_size': 64, 'generalized': False, 'cls_train_steps': 22},
      {'dataset': 'AWA2', 'latent_size': 64, 'generalized': True, 'cls_train_steps': 50},
      {'dataset': 'AWA2', 'latent_size': 64, 'generalized': False, 'cls_train_steps': 39},
      ]

##################################
# change some hyperparameters here
##################################
hyperparameters['dataset'] = args.dataset
hyperparameters['latent_size']= args.latent_size
hyperparameters['generalized']= args.generalized

hyperparameters['cls_train_steps'] = [x['cls_train_steps']  for x in cls_train_steps
                                        if all([hyperparameters['dataset']==x['dataset'],
                                        hyperparameters['latent_size']==x['latent_size'],
                                        hyperparameters['generalized']==x['generalized'] ])][0]

print('***')
print(hyperparameters['cls_train_steps'] )
if hyperparameters['generalized']:
    if hyperparameters['num_shots']==0:
        hyperparameters['samples_per_class'] = {'CUB': (200, 0, 400, 0), 'SUN': (200, 0, 400, 0),
                                'APY': (200, 0,  400, 0), 'AWA1': (200, 0, 400, 0),
                                'AWA2': (200, 0, 400, 0), 'FLO': (200, 0, 400, 0)}
    else:
        hyperparameters['samples_per_class'] = {'CUB': (200, 0, 200, 200), 'SUN': (200, 0, 200, 200),
                                                    'APY': (200, 0, 200, 200), 'AWA1': (200, 0, 200, 200),
                                                    'AWA2': (200, 0, 200, 200), 'FLO': (200, 0, 200, 200)}
else:
    if hyperparameters['num_shots']==0:
        hyperparameters['samples_per_class'] = {'CUB': (0, 0, 200, 0), 'SUN': (0, 0, 200, 0),
                                                    'APY': (0, 0, 200, 0), 'AWA1': (0, 0, 200, 0),
                                                    'AWA2': (0, 0, 200, 0), 'FLO': (0, 0, 200, 0)}
    else:
        hyperparameters['samples_per_class'] = {'CUB': (0, 0, 200, 200), 'SUN': (0, 0, 200, 200),
                                                    'APY': (0, 0, 200, 200), 'AWA1': (0, 0, 200, 200),
                                                    'AWA2': (0, 0, 200, 200), 'FLO': (0, 0, 200, 200)}


model = Model( hyperparameters)
model.to(hyperparameters['device'])




model.test()




print(time.strftime('ending time:%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
print("dataset", args.dataset)
print(hyperparameters['classifier'])
print("**********END*******************")

