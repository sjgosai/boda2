import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt
from warnings import warn


class FastSeqProp(nn.Module):
    def __init__(self,
                 num_sequences=1,
                 seq_len=200, 
                 padding_len=0,
                 upPad_DNA=None,
                 downPad_DNA=None,
                 vocab_list=['A','G','T','C'],
                 seed=None,
                 **kwargs):
        super(FastSeqProp, self).__init__()
        self.num_sequences = num_sequences
        self.seq_len = seq_len  
        self.padding_len = padding_len
        self.upPad_DNA = upPad_DNA
        self.downPad_DNA = downPad_DNA
        self.vocab_list = vocab_list
        self.seed = seed
        
        self.vocab_len = len(vocab_list)       
        self.noise_factor = 0
        self.softmaxed_logits = None

        self.create_paddingTensors()      
        self.set_seed()
        
        #initialize the trainable logits     
        self.create_differentiable_input_logits(one_hot=True)      
        
        #instance normalization layer
        self.instance_norm = nn.InstanceNorm1d(num_features=self.vocab_len, affine=True)
        
    def forward(self):
        #scaled softmax relaxation
        normalized_logits = self.instance_norm(self.differentiable_logits) + \
             self.noise_factor*torch.randn_like(self.differentiable_logits)
        softmaxed_logits = F.softmax(normalized_logits, dim=1)
        #save attributes without messing the backward graph
        self.softmaxed_logits = softmaxed_logits
        self.padded_softmaxed_logits = self.pad(softmaxed_logits)
        #sample
        nucleotide_probs = Categorical(torch.transpose(softmaxed_logits, 1, 2))
        sampled_idxs = nucleotide_probs.sample()
        sampled_nucleotides_T = F.one_hot(sampled_idxs, num_classes=self.vocab_len)        
        sampled_nucleotides = torch.transpose(sampled_nucleotides_T, 1, 2)
        sampled_nucleotides = sampled_nucleotides - softmaxed_logits.detach() + softmaxed_logits  #ST estimator trick
        softmaxed_logits, sampled_nucleotides = self.pad(softmaxed_logits), self.pad(sampled_nucleotides)
        return softmaxed_logits, sampled_nucleotides
    
    def create_differentiable_input_logits(self, one_hot=True):
        size = (self.num_sequences, self.vocab_len, self.seq_len)
        if one_hot:
            differentiable_logits = np.zeros(size)
            for seqIdx in range(self.num_sequences):
                for step in range(self.seq_len):
                    randomNucleotide = np.random.randint(self.vocab_len)
                    differentiable_logits[seqIdx, randomNucleotide, step] = 1       
            self.differentiable_logits = nn.Parameter(torch.tensor(differentiable_logits, dtype=torch.float))  
        else:
            self.differentiable_logits = nn.Parameter(torch.rand(size))
         
    def set_seed(self):
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
                
    def create_paddingTensors(self):
        assert self.padding_len >= 0 and type(self.padding_len) == int, 'Padding must be a nonnegative integer'
        if self.padding_len > 0:
            assert self.padding_len <= (len(self.upPad_DNA) + len(self.downPad_DNA)), 'Not enough padding available'
            upPad_logits, downPad_logits = self.dna2tensor(self.upPad_DNA), \
                                         self.dna2tensor(self.downPad_DNA)
            upPad_logits, downPad_logits = upPad_logits[:,-self.padding_len//2 + self.padding_len%2:], \
                                         downPad_logits[:,:self.padding_len//2 + self.padding_len%2]
            upPad_logits, downPad_logits = upPad_logits.repeat(self.num_sequences, 1, 1), \
                                         downPad_logits.repeat(self.num_sequences, 1, 1)
            self.register_buffer('upPad_logits', upPad_logits)
            self.register_buffer('downPad_logits', downPad_logits)
        else:
            self.upPad_logits, self.downPad_logits = None, None

    def pad(self, tensor):
        if self.padding_len > 0:
            padded_tensor = torch.cat([ self.upPad_logits, tensor, self.downPad_logits], dim=2)
            return padded_tensor
        else: 
            return tensor
    
    def dna2tensor(self, sequence_str):
        seq_tensor = np.zeros((self.vocab_len, len(sequence_str)))
        for letterIdx, letter in enumerate(sequence_str):
            seq_tensor[self.vocab_list.index(letter), letterIdx] = 1
        seq_tensor = torch.Tensor(seq_tensor)
        return seq_tensor
    
    def optimize(self, predictor, loss_fn, steps=20, learning_rate=0.5, 
                 step_print=5, lr_scheduler=True, noise_factor=0):
        self.noise_factor = noise_factor
        if lr_scheduler:
            etaMin = 0.000001
        else:
            etaMin = learning_rate
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=etaMin)      
        print('-----Initial logits-----')
        print(self.pad(self.differentiable_logits).detach().numpy())
        print('-----Training steps-----')
        loss_hist  = []
        for step in range(1, steps+1):
            optimizer.zero_grad()
            softmaxed_logits, sampled_nucleotides = self()
            predictions = predictor(sampled_nucleotides)
            loss = loss_fn(predictions)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_hist.append(loss.item())
            self.noise_factor = self.noise_factor / np.sqrt(step)
            if step % step_print == 0:
                print(f'step: {step}, loss: {round(loss.item(),6)}, learning_rate: {scheduler.get_last_lr()}, noise factor: {self.noise_factor}')                      
        self.noise_factor = 0
        print('-----Final distribution-----')
        print(self.padded_softmaxed_logits.detach().numpy())
        self.loss_hist = loss_hist
        plt.plot(loss_hist)
        plt.xlabel('Steps')
        vert_label=plt.ylabel('Loss')
        vert_label.set_rotation(90)
        plt.show()
    
    def generate(self, padded=True):
        if self.softmaxed_logits == None:
            warn('The model hasn\'t been trained yet')
            return self.pad(self.differentiable_logits.detach())
        else:
            nucleotide_probs = Categorical(torch.transpose(self.softmaxed_logits, 1, 2))
            sampled_idxs = nucleotide_probs.sample()
            sampled_nucleotides_T = F.one_hot(sampled_idxs, num_classes=self.vocab_len)        
            sampled_nucleotides = torch.transpose(sampled_nucleotides_T, 1, 2)
            if padded:
                sampled_nucleotides = self.pad(sampled_nucleotides)
            return sampled_nucleotides.detach()


#--------------------------- EXAMPLE ----------------------------------------
# if __name__ == '__main__':
#     from FastSeqProp_utils import first_token_rewarder, neg_reward_loss
#     import sys
#     sys.path.insert(0, '/Users/castrr/Documents/GitHub/boda2/')    #edit path to boda2
#     from boda.common import constants     
#     np.set_printoptions(precision=2)    # for shorter display of np arrays
    
#     model = FastSeqProp(num_sequences=1,
#                         seq_len=5,
#                         padding_len=2,
#                         upPad_DNA=constants.MPRA_UPSTREAM,
#                         downPad_DNA=constants.MPRA_DOWNSTREAM,
#                         vocab_list=constants.STANDARD_NT,
#                         seed=None)
#     model.optimize(predictor=first_token_rewarder,
#                     loss_fn=neg_reward_loss,
#                     steps=100,
#                     learning_rate=0.5,
#                     step_print=20,
#                     lr_scheduler=True,
#                     noise_factor=0.005)
#     sample_example = model.generate()
    
#     print('-----Sample example-----')
#     print(sample_example.numpy())