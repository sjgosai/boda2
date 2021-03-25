import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

import sys
sys.path.insert(0, '/Users/castrr/Documents/GitHub/boda2/')    #edit path to boda2
from boda.model.mpra_basset import MPRA_Basset
from boda.data.BODA2_DataModule import BODA2_DataModule
%load_ext tensorboard


def main(args):
    datamodule = BODA2_DataModule(**vars(args))
    datamodule.setup()
    model = MPRA_Basset(**vars(args))
    logger = TensorBoardLogger('model_logs', name='MPRAbasset_logs', log_graph=True) 
    if args.only_last_layer:
        print('Training only last layer')
        model.basset_net.freeze() 
    else:
        print('Training all layers')
        model.basset_net.unfreeze()
        
    num_gpus = torch.cuda.device_count()
    if model.scheduler:
        lr_monitor = LearningRateMonitor(logging_interval='step')
        trainer = pl.Trainer(gpus=num_gpus, max_epochs=model.epochs, progress_bar_refresh_rate=20, logger=logger, callbacks=[lr_monitor])
    else:
        trainer = pl.Trainer(gpus=num_gpus, max_epochs=model.epochs, progress_bar_refresh_rate=20, logger=logger)
    
    trainer.fit(model, datamodule)
    trainer.test()
    %tensorboard --logdir model_logs/MPRAbasset_logs
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MPRA_Basset', add_help=False)
    parser.add_argument('--dataFile_path', type=str,
                        default='/Users/castrr/Documents/GitHub/boda2/boda/common/BODA.MPRA.txt')
    parser.add_argument('--basset_weights_path', type=str,
                        default='/Users/castrr/Documents/GitHub/boda2/boda/common/my-model.epoch_5-step_19885.pkl')
    parser.add_argument('--sequenceColumn', type=str, default='nt.sequence')
    parser.add_argument('--activityColumns', type=list,  default=['K562', 'HepG2', 'SKNSH'])
    parser.add_argument('--only_last_layer', type=bool, default=True)
    
    parser = MPRA_Basset.add_model_specific_args(parser)
    parser = BODA2_DataModule.add_data_specific_args(parser)
    args = parser.parse_args()
    print(f'Parser arguments: {vars(args)}')
    main(args)
