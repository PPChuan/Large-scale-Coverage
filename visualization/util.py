import os

import torch


def load_weights(pth_path, model):
    assert os.path.isfile(pth_path)
    ckpt = torch.load(pth_path)
    model_name = 'simclr'
    for k, _ in ckpt.items():
        if k.startswith('encoder.'):
            model_name = 'simsiam'
            break
        elif k.startswith('encoder_q.'):
            model_name = 'moco+byol'
            break
    new_state_dict = {}
    if model_name == 'simclr':
        for k, v in ckpt.items():
            newk = k
            new_state_dict[newk] = v
    elif model_name == 'simsiam':
        for k, v in ckpt.items():
            if k.startswith('encoder.') and not k.startswith('encoder.fc'):
                newk = k.replace('encoder.', '')
                new_state_dict[newk] = v
    elif model_name == 'moco+byol':
        for k, v in ckpt.items():
            if k.startswith('encoder_q.'):
                newk = k.replace('encoder_q.', '')
                new_state_dict[newk] = v
    else:
        raise Exception('Error: From util.py load_weights(function), invalid model.')
    del ckpt
    msg = model.load_state_dict(new_state_dict, strict=False)
    assert msg.missing_keys == []
