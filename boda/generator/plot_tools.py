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

from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

import dmslogo
import palettable
import dmslogo.colorschemes
from dmslogo.colorschemes import CBPALETTE

import imageio

def motif_str_to_counts(motif_str, pseudocounts=1.0):
    """
    Convert a motif string to a tensor of motif counts.

    Args:
        motif_str (str): The motif string, where each row represents a position and columns represent nucleotide frequencies.
        pseudocounts (float): Pseudocount value added to motif counts.

    Returns:
        torch.Tensor: Tensor of motif counts.
    """
    motif = torch.tensor( [ list([ float(y) for y in x.split() ]) for x in motif_str.split('\n') ] )
    return motif.add(pseudocounts)

def counts_to_ppm(in_tensor):
    """
    Convert motif counts to a Position Probability Matrix (PPM).

    Args:
        in_tensor (torch.Tensor): Input tensor of motif counts.

    Returns:
        torch.Tensor: Position Probability Matrix (PPM).
    """
    motif = in_tensor.div( in_tensor.sum(0) )
    return motif

def ppm_to_pwm(in_tensor,bkg=[0.25,0.25,0.25,0.25]):
    """
    Convert a Position Probability Matrix (PPM) to a Position Weight Matrix (PWM).

    Args:
        in_tensor (torch.Tensor): Input tensor of Position Probability Matrix (PPM).
        bkg (list): Background nucleotide frequencies.

    Returns:
        torch.Tensor: Position Weight Matrix (PWM).
    """
    motif = in_tensor.div(torch.tensor(bkg).unsqueeze(1)).log2()
    return motif

def ppm_to_U(in_tensor):
    """
    Convert a Position Probability Matrix (PPM) to the uncertainty (U) of each position.

    Args:
        in_tensor (torch.Tensor): Input tensor of Position Probability Matrix (PPM).

    Returns:
        torch.Tensor: Uncertainty (U) of each position.
    """
    return (in_tensor.log2() * in_tensor).sum(0).mul(-1.)

def ppm_to_IC(in_tensor):
    """
    Convert a Position Probability Matrix (PPM) to the information content (IC) matrix.

    Args:
        in_tensor (torch.Tensor): Input tensor of Position Probability Matrix (PPM).

    Returns:
        torch.Tensor: Information content (IC) matrix.
    """
    return in_tensor * ppm_to_U( in_tensor ).mul(-1.).add(2.)

def PWM_to_filter(motif_str):
    """
    Convert a Position Weight Matrix (PWM) from a motif string to a filter.

    Args:
        motif_str (str): The motif string representing a Position Weight Matrix (PWM).

    Returns:
        torch.Tensor: Filter generated from PWM.
    """
    motif = torch.tensor( [ list([ float(y) for y in x.split() ]) for x in motif_str.split('\n') ] )
    motif = motif.add(1.).div( motif.add(1.).sum(0) )
    motif = motif.div( motif.add(-1).mul(-1) ).log().clamp(-5,5)
    motif = torch.randn_like(motif).mul(1e-3).add(motif)
    motif = motif.add(-motif.mean(0))
    return motif

def counts_to_filter(in_tensor):
    """
    Convert motif counts to a filter using Position Probability Matrix (PPM) and Position Weight Matrix (PWM) conversion.

    Args:
        in_tensor (torch.Tensor): Input tensor of motif counts.

    Returns:
        torch.Tensor: Filter generated from motif counts.
    """
    motif = ppm_to_pwm( counts_to_ppm( in_tensor ) )
    return motif.unsqueeze(0)

def tensor_to_pandas(in_tensor, tokens=constants.STANDARD_NT, colors=['green','orange','red','blue']):
    """
    Convert a tensor of motif scores to a pandas DataFrame for visualization.

    Args:
        in_tensor (torch.Tensor): Input tensor of motif scores.
        tokens (list): List of nucleotide symbols.
        colors (list): List of colors for visualization.

    Returns:
        pd.DataFrame: Pandas DataFrame containing motif score data.
    """
    data = []
    my_array = in_tensor.cpu().numpy()
    for nt, score_vec,color in zip(tokens,[ sub_a for sub_a in my_array ],colors):
         _ = [ data.append([j,nt,score,color]) for j,score in enumerate(score_vec) ]
    return pd.DataFrame(data=data,columns=['site', 'letter', 'height','color'])

def logits_to_dms(in_tensor, target='dms_motif.png', ax=None):
    """
    Convert logit scores to a DMS motif logo and save it as an image.

    Args:
        in_tensor (torch.Tensor): Input tensor of logit scores.
        target (str): Path to save the DMS motif logo image.
        ax: Matplotlib axis to use for plotting.

    Returns:
        matplotlib.figure.Figure: Generated figure of the DMS motif logo.
        matplotlib.axes._axes.Axes: Matplotlib axis containing the logo.
    """
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

def samples_to_dms(in_tensor, target='dms_motif.png', ax=None):
    """
    Convert sequence samples to a DMS motif logo and save it as an image.

    Args:
        in_tensor (torch.Tensor): Input tensor of sequence samples.
        target (str): Path to save the DMS motif logo image.
        ax: Matplotlib axis to use for plotting.

    Returns:
        matplotlib.figure.Figure: Generated figure of the DMS motif logo.
        matplotlib.axes._axes.Axes: Matplotlib axis containing the logo.
    """
    motif = in_tensor.sum(dim=0)
    motif = counts_to_ppm( motif )
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

def matrix_to_dms(in_tensor, ax=None, y_max=2, fontaspect=.65, widthscale=0.8,
                  axisfontscale=0.6, heightscale=0.5):
    """
    Convert a matrix of motif values to a DMS motif logo and visualize it.

    Args:
        in_tensor (torch.Tensor): Input tensor of motif values.
        ax: Matplotlib axis to use for plotting.
        y_max (float): Maximum height for the logo.
        fontaspect (float): Font aspect ratio for plotting.
        widthscale (float): Width scale for plotting.
        axisfontscale (float): Axis font scale for plotting.
        heightscale (float): Height scale for plotting.

    Returns:
        matplotlib.figure.Figure: Generated figure of the DMS motif logo.
        matplotlib.axes._axes.Axes: Matplotlib axis containing the logo.
    """
    motif = tensor_to_pandas(in_tensor)
    fig, ax = dmslogo.draw_logo(data=motif,
                                x_col='site',
                                letter_col='letter',
                                letter_height_col='height',
                                color_col='color',
                                ax=ax,
                                fixed_ymax=y_max,
                                fontaspect=fontaspect,
                                widthscale=widthscale,
                                axisfontscale=axisfontscale,
                                heightscale=heightscale)
    return fig, ax

def dms_video(theta_tensor, energy_tensor, target='my_motif.mp4'):
    """
    Generate a video showing the optimization process of a DMS motif.

    Args:
        theta_tensor (torch.Tensor): Input tensor of optimization theta values.
        energy_tensor (torch.Tensor): Input tensor of energy values during optimization.
        target (str): Path to save the generated video.

    Returns:
        list: List of images used in the video.
    """
    images = []
    writer = imageio.get_writer(target, fps=25)
    hold_range = torch.arange(energy_tensor.shape[0])
    seq_len = theta_tensor.shape[-1]
    fig_len = max(1,seq_len//50)*12
    min_e = energy_tensor.min()
    max_e = energy_tensor.max()
    for i, a_theta in enumerate(tqdm([ x for x in theta_tensor ])):
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
