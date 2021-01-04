import argparse
import sys
import os
import shutil

import torch
from pytorch_lightning import Trainer

import cifar10

def main(args):
    data = cifar10.MyDataModule(**vars(args))
    
    if args.pretrained_model is None:
        model = cifar10.MyNet(**vars(args))
    else:
        model = model_fn(args.pretrained_model)
        
    trainer = Trainer.from_argparse_args(args)
    
    trainer.fit(model, data)
    
    _save_model(trainer, args.sagemaker_checkpoint)
    
    return None

def _save_model(trainer, model_dir, inference_script_dir=None):
    trainer.save_checkpoint(os.path.join(model_dir, 'model.ckpt'))
    os.makedirs(os.path.join(model_dir,'code'), exist_ok=True)
    if inference_script_dir is None:
        script_dir = os.path.dirname(sys.argv[0])
    else:
        script_dir = inference_script_dir
    shutil.copy(os.path.join(script_dir,'./inference.py'), 
                os.path.join(model_dir,'code'))
    return None

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CIFAR10", add_help=False)
    parser.add_argument('--sagemaker_checkpoint', type=str, default=os.getenv('SM_MODEL_DIR'))
    parser.add_argument('--pretrained_model', type=str, default=os.getenv('SM_CHANNEL_MODEL'))
    
    parser = cifar10.MyDataModule.add_data_specific_args(parser)
    parser = cifar10.MyNet.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--help','-h',action='help')
    
    args = parser.parse_args()
    print(vars(args))
    
    inference_script = os.path.join(os.path.dirname(sys.argv[0]),'./inference.py')
    assert os.path.isfile(inference_script), "inference.py should be in same dir as main.py"
    
    main(args)
