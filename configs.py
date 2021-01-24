labels = ['t_shirt_top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boots']
epochs = 1000
batch_size =128
model_configs = {'with_batch_norm': {'batch_norm': True,  'lr':0.01},
     'with_weight_decay': {'weight_decay': 0.001, 'lr':0.01},
     'with_dropout': {'dropout': 0.2, 'lr':0.01},
     'no_regularization': {'lr':0.01}
     }


