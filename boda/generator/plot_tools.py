import sys
import os
import subprocess
import tarfile
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import constants

import matplotlib.pyplot as plt
import seaborn as sns

import dmslogo
import palettable
import dmslogo.colorschemes
from dmslogo.colorschemes import CBPALETTE

import imageio

def motif_str_to_counts(motif_str, pseudocounts=1.0):
    motif = torch.tensor( [ list([ float(y) for y in x.split() ]) for x in motif_str.split('\n') ] )
    return motif.add(pseudocounts)

def counts_to_ppm(in_tensor):
    motif = in_tensor.div( in_tensor.sum(0) )
    return motif

def ppm_to_pwm(in_tensor,bkg=[0.25,0.25,0.25,0.25]):
    motif = in_tensor.div(torch.tensor(bkg).unsqueeze(1)).log2()
    return motif

def ppm_to_U(in_tensor):
    return (in_tensor.log2() * in_tensor).sum(0).mul(-1.)

def ppm_to_IC(in_tensor):
    return in_tensor * ppm_to_U( in_tensor ).mul(-1.).add(2.)

def PWM_to_filter(motif_str):
    motif = torch.tensor( [ list([ float(y) for y in x.split() ]) for x in motif_str.split('\n') ] )
    motif = motif.add(1.).div( motif.add(1.).sum(0) )
    motif = motif.div( motif.add(-1).mul(-1) ).log().clamp(-5,5)
    motif = torch.randn_like(motif).mul(1e-3).add(motif)
    motif = motif.add(-motif.mean(0))
    return motif

def counts_to_filter(in_tensor):
    motif = ppm_to_pwm( counts_to_ppm( in_tensor ) )
    return motif.unsqueeze(0)

def tensor_to_pandas(in_tensor, tokens=constants.STANDARD_NT, colors=['green','orange','red','blue']):
    data = []
    my_array = in_tensor.cpu().numpy()
    for nt, score_vec,color in zip(tokens,[ sub_a for sub_a in my_array ],colors):
         _ = [ data.append([j,nt,score,color]) for j,score in enumerate(score_vec) ]
    return pd.DataFrame(data=data,columns=['site', 'letter', 'height','color'])

def logits_to_dms(in_tensor, target='dms_motif.png', ax=None):
    motif = F.softmax(in_tensor, dim=0)
    motif = ppm_to_IC( motif )
    motif = tensor_to_pandas(motif)
    fig, ax = dmslogo.draw_logo(data=motif,
                                x_col='site',
                                letter_col='letter',
                                letter_height_col='height',
                                color_col='color',
                                fixed_ymax=2.0,
                                ax=ax)
    if target is not None:
        fig.savefig(target,dpi=400)
    return fig, ax

def dms_video(theta_tensor, energy_tensor, target='my_motif.mp4'):
    images = []
    writer = imageio.get_writer(target, fps=25)
    hold_range = torch.arange(energy_tensor.shape[0])
    seq_len = theta_tensor.shape[-1]
    fig_len = max(1,seq_len//50)*12
    min_e = energy_tensor.min()
    max_e = energy_tensor.max()
    for i, a_theta in enumerate([ x for x in theta_tensor ]):
        #_ = tensor_to_dms(a_theta, target='dms_motif.png')
        
        fig = plt.figure(figsize=(fig_len,8))
        axes = fig.subplots(nrows=2)
        axes[1].plot(hold_range[:i].cpu().numpy(), energy_tensor[:i].cpu().numpy())
        axes[1].set_xlim(0, energy_tensor.shape[0])
        axes[1].set_ylim(min_e, max_e)
        axes[1].set_ylabel('Potential Energy')
        axes[1].set_xlabel('Iteration')
        _ = logits_to_dms(a_theta, target=None,ax=axes[0])
        fig.savefig('temp_line.png',dpi=100)
        plt.close()
        
        #two_pngs('dms_motif.png','temp_line.png','compound_image.png',dpi=400)
        
        images.append( imageio.imread('temp_line.png') )
    for an_image in images:
        writer.append_data(an_image)
    writer.close()
    return images
