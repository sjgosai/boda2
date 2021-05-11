import json
import logging
import os
import copy

import torch
from pytorch_lightning import Trainer

import cifar10

JSON_CONTENT_TYPE = 'application/json'

logger = logging.getLogger(__name__)


def model_fn(model_dir):
    # PTL friendly version
    #return cifar10.MyNet.load_from_checkpoint(checkpoint_path=os.path.join(model_dir,'model.ckpt'))
    # Hack to deal with setup
    checkpoint = torch.load(os.path.join(model_dir,'model.ckpt'))
    new_model = cifar10.MyNet(**checkpoint['hyper_parameters'])
    new_model.setup('train')
    new_model.load_state_dict(checkpoint['state_dict'])
    new_model.eval()
    return new_model

def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):
    logger.info('Deserializing input data.')
    if content_type == JSON_CONTENT_TYPE:
        input_data = json.loads(serialized_input_data)
    else:
        raise NotImplementedError(f"{content_type} is an unsuported ContentType for content_type, use {JSON_CONTENT_TYPE}")
    return input_data

def predict_fn(input_data, model):
    prediction_output = copy.copy(input_data)
    logger.info('Generating image class predictions.')
    request = torch.tensor(input_data['images'],dtype=torch.float32)
    response = [ model(x.unsqueeze(0).to(model.device)).detach().cpu().numpy().tolist()
                 for x in request ]
    prediction_output['predictions'] = response
    return prediction_output
    
def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    logger.info('Serializing class predictions.')
    if accept == JSON_CONTENT_TYPE:
        output_data = json.dumps(prediction_output)
    else:
        raise NotImplementedError(f"{content_type} is an unsuported ContentType for accept, use {JSON_CONTENT_TYPE}")
    return output_data, accept