import os
import time
import yaml
import shutil
import argparse
import tarfile
import tempfile
import random

import torch
from pytorch_lightning import Trainer

import boda

def main(args):
    data_module = getattr(boda.data, args['Main args'].data_module)
    model_module= getattr(boda.model, args['Main args'].model_module)
    graph_module= getattr(boda.graph, args['Main args'].graph_module)

    data = data_module(**vars(data_module.process_args(args)))
    model= model_module(**vars(model_module.process_args(args)))

    model.__class__ = type(
        'BODA_module',
        (model_module,graph_module),
        vars(graph_module.process_args(args))
    )

    os.makedirs('/tmp/output/artifacts', exist_ok=True)
    trainer = Trainer.from_argparse_args(args['pl.Trainer'])
    
    trainer.fit(model, data)
    
    _save_model(data_module, model_module, graph_module, 
                model, trainer, args)
    
def _save_model(data_module, model_module, graph_module, 
                model, trainer, args):
    local_dir = args['pl.Trainer'].default_root_dir
    save_dict = {
        'data_module'  : data_module.__name__,
        'data_hparams' : data_module.process_args(args),
        'model_module' : model_module.__name__,
        'model_hparams': model_module.process_args(args),
        'graph_module' : graph_module.__name__,
        'graph_hparams': graph_module.process_args(args),
        'model_state_dict': model.state_dict(),
        'timestamp'    : time.strftime("%Y%m%d_%H%M%S"),
        'random_tag'   : random.randint(100000,999999)
    }
    torch.save(save_dict, os.path.join(local_dir,'torch_checkpoint.pt'))
    
    filename=f'model_artifacts__{save_dict["timestamp"]}__{save_dict["random_tag"]}.tar.gz'
    with tempfile.TemporaryDirectory() as tmpdirname:
        with tarfile.open(os.path.join(tmpdirname,filename), 'w:gz') as tar:
            tar.add(local_dir,arcname='artifacts')

        if 'gs://' in args['Main args'].artifact_path:
            subprocess.check_call(
                ['gsutil', 'cp', os.path.join(tmpdirname,filename), args['Main args'].artifact_path]
            )
        else:
            os.makedirs(args['Main args'].artifact_path, exist_ok=True)
            shutil.copy(os.path.join(tmpdirname,filename), args['Main args'].artifact_path)
    
def model_fn(model_dir):
    checkpoint = torch.load(os.path.join(model_dir,'torch_checkpoint.pt'))
    model_module = getattr(boda, checkpoint['model_module'])
    model        = model_module(**checkpoint['model_hparams'])
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded model from {checkpoint["timestamp"]}')
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="BODA trainer", add_help=False)
    group = parser.add_argument_group('Main args')
    group.add_argument('--data_module', type=str, required=True, help='BODA data module to process dataset.')
    group.add_argument('--model_module',type=str, required=True, help='BODA model module to fit dataset.')
    group.add_argument('--graph_module',type=str, required=True, help='BODA graph module to define computations.')
    group.add_argument('--artifact_path', type=str, default='/opt/ml/checkpoints/', help='Path where model artifacts are deposited.')
    group.add_argument('--pretrained_weights', type=str, help='Pretrained weights.')
    known_args, leftover_args = parser.parse_known_args()
    
    Data  = getattr(boda.data,  known_args.data_module)
    Model = getattr(boda.model, known_args.model_module)
    Graph = getattr(boda.graph, known_args.graph_module)
    
    parser = Data.add_data_specific_args(parser)
    parser = Model.add_model_specific_args(parser)
    parser = Graph.add_graph_specific_args(parser)
    
    known_args, leftover_args = parser.parse_known_args()
    
    parser = Data.add_conditional_args(parser, known_args)
    parser = Model.add_conditional_args(parser, known_args)
    parser = Graph.add_conditional_args(parser, known_args)
    
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--help', '-h', action='help')
    args = parser.parse_args()
    
    args = boda.common.utils.organize_args(parser, args)
    
    main(args)