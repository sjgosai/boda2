import argparse
import torch

def str2bool(v):
    """Pulled from https://stackoverflow.com/a/43357954
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
'''
Dummy predictor for examples
Reward the percentage of ones in first token
'''
def first_token_rewarder(sequences):
    weights = torch.zeros(sequences.shape)
    weights[:,0,:] = 1
    rewards = (weights * sequences).sum(2).sum(1) / sequences.shape[2]
    rewards = rewards.view(-1, 1)
    return rewards

'''
Dummy loss for examples
For maximizing avg reward
'''
def neg_reward_loss(x):
    return -torch.sum(x)