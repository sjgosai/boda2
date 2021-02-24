import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt

class NUTS(nn.Module):
    def __init__(self,
                 fitness_fn=None,
                 num_sequences=1,
                 seq_len=200, 
                 padding_len=0,
                 upPad_DNA=None,
                 downPad_DNA=None,
                 vocab_list=['A','G','T','C'],
                 seed=None,
                 temperature=1,
                 kinetic_scale_factor=1,
                 **kwargs):
        super(NUTS, self).__init__()
        self.fitness_fn = fitness_fn
        self.num_sequences = num_sequences
        self.seq_len = seq_len  
        self.padding_len = padding_len
        self.upPad_DNA = upPad_DNA
        self.downPad_DNA = downPad_DNA
        self.vocab_list = vocab_list
        self.seed = seed
        self.temperature = temperature
        self.kinetic_scale_factor = kinetic_scale_factor
        self.Delta_max = 1000
        self.delta = 0.65
        self.fitness_hist = []
        
        self.softmax = nn.Softmax(dim=1)
        self.grad = torch.autograd.grad
        self.vocab_len = len(vocab_list)       
        self.create_paddingTensors()      
        self.set_seed()
        
        # initialize theta and r
        self.initialize_theta(one_hot=True)
        #self.r = torch.randn_like(self.theta)
        self.register_buffer('r', torch.randn_like(self.theta))

    def set_seed(self):
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
                
    #this could be implemented outside the class
    def dna2tensor(self, sequence_str):
        seq_tensor = np.zeros((self.vocab_len, len(sequence_str)))
        for letterIdx, letter in enumerate(sequence_str):
            seq_tensor[self.vocab_list.index(letter), letterIdx] = 1
        seq_tensor = torch.Tensor(seq_tensor)
        return seq_tensor
    
    #this could be implemented outside the class
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
    
    #this could be implemented outside the class
    def pad(self, tensor):
        if self.padding_len > 0:
            padded_tensor = torch.cat([ self.upPad_logits, tensor, self.downPad_logits], dim=2)
            return padded_tensor
        else: 
            return tensor
    
    #this could be implemented outside the class   
    def initialize_theta(self, one_hot=True):
        size = (self.num_sequences, self.vocab_len, self.seq_len)
        if one_hot:
            theta = np.zeros(size)
            for seqIdx in range(self.num_sequences):
                for step in range(self.seq_len):
                    random_token = np.random.randint(self.vocab_len)
                    theta[seqIdx, random_token, step] = 1      
            self.register_buffer('theta', torch.tensor(theta, dtype=torch.float))
        else:
            self.register_buffer('theta', self.softmax(torch.rand(size)))
            
    def L_fn(self, theta):
        softmaxed_theta = self.softmax(theta / self.temperature)
        nucleotide_probs = Categorical(torch.transpose(softmaxed_theta, 1, 2))
        sampled_idxs = nucleotide_probs.sample()
        sampled_nucleotides_T = F.one_hot(sampled_idxs, num_classes=self.vocab_len)        
        sampled_nucleotides = torch.transpose(sampled_nucleotides_T, 1, 2)
        sampled_nucleotides = sampled_nucleotides - softmaxed_theta.detach() + softmaxed_theta  #ST estimator trick
        sampled_nucleotides = self.pad(sampled_nucleotides)
        return -self.fitness_fn(sampled_nucleotides).sum()
    
    def p_fn(self, theta=None, r=None, L=None):
        if theta is not None:
            return torch.exp(self.L_fn(theta) - r.pow(2).sum()/2 * self.kinetic_scale_factor).item()
        elif L is not None:
            return torch.exp(L - r.pow(2).sum()/2 * self.kinetic_scale_factor).item()
    
    def leapfrog(self, theta, r, epsilon):
        # make theta a leaf
        theta.requires_grad_()    
        # compute grad
        L = self.L_fn(theta)
        L_grad = self.grad(L, theta, retain_graph=False)[0]        
        # momentum half step
        r_prime = r + 0.5 * epsilon * L_grad        
        # position full step
        theta_prime = theta + epsilon * r_prime       
        # compute new grad
        L_prime = self.L_fn(theta_prime)
        L_grad_prime = self.grad(L_prime, theta_prime, retain_graph=False)[0]        
        # momentum half step
        r_prime = r_prime + 0.5 * epsilon * L_grad_prime 
        p_prime = self.p_fn(r=r_prime, L=L_prime)
        # return non-leaf
        theta_prime = theta_prime.detach()
        theta.requires_grad = False
        return theta_prime, r_prime, L_prime.item(), L_grad_prime, p_prime

    def run_HMC(self, num_steps, epsilon=1):
        self.fitness_hist.append(self.fitness_fn(self.pad(self.theta)))
        #print(self.theta)
        alpha_list = []
        for step in range(num_steps):
            r_0 = torch.randn_like(self.r)
            p = self.p_fn(theta=self.theta, r=r_0)
            theta_prime, r_prime, L_prime, _, p_prime = self.leapfrog(self.theta, self.r, epsilon)
            alpha = min(1, p_prime/(p + 1e-20))
            random_prob = float(np.random.rand(1))
            alpha_list.append(alpha)
            if alpha >= random_prob:
                self.theta, self.r = theta_prime, r_prime
                self.fitness_hist.append(-L_prime)            
        #print(self.softmax(self.theta))
        plt.plot(self.fitness_hist)
        plt.ylim(-0.01,1.01)
        plt.title('Fitness')
        plt.show()
        print(f'Average acceptance probability: {np.mean(alpha_list)}')
        
    # adapted from https://github.com/mfouesneau/NUTS
    def find_reasonable_epsilon(self):
        p = self.p_fn(theta=self.theta, r=self.r)
        epsilon = 1
        _, r_prime, _, L_grad_prime, p_prime = self.leapfrog(self.theta, self.r, epsilon)
        # check the initial step size does not yield infinite values of p or the grad
        k = 1
        while np.isinf(p_prime) or torch.isinf(L_grad_prime).any():   
            k *= 0.5
            epsilon = k * epsilon
            _, _, _, L_grad_prime, p_prime = self.leapfrog(self.theta, self.r, epsilon)              
        # set a = 2*I[p_prime/p > 0.5] - 1
        a = 1. if p_prime/(p + 1e-12) > 0.5 else -1.
        while (np.log(p_prime + 1e-12) - np.log(p + 1e-12)) * a > np.log(0.5) * a:
            epsilon = epsilon * (2. ** a)
            _, _, _, _, p_prime = self.leapfrog(self.theta, self.r, epsilon)
        #print(f'Initial reasonable epsilon = {epsilon}')
        return epsilon
    
    def stop_indicator(self, theta_minus, r_minus, theta_plus, r_plus):
        theta_diff = theta_plus - theta_minus
        s = (torch.mul(theta_diff, r_plus).sum() >= 0) and (torch.mul(theta_diff, r_minus).sum() >= 0)
        if s == False: print('Stopped by U-turn')
        return 1 * s.item()
        
    def build_tree(self, theta, r, u, v, j, epsilon, theta_0, r_0):
        if j == 0:   
            theta_prime, r_prime, L_prime, L_grad_prime, p_prime = self.leapfrog(theta, r, v * epsilon)
            n_prime = 1 if u <= p_prime else 0
            s_prime = 1 if np.log(u) < (np.log(p_prime + 1e-12) + self.Delta_max) else 0
            p_0 = self.p_fn(theta=theta_0, r=r_0)
            alpha = min(1, p_prime/(p_0 + 1e-12))
            self.fitness_hist.append(-L_prime)
            return theta_prime, r_prime, theta_prime, r_prime, theta_prime, n_prime, s_prime, alpha, 1.
        else:
            theta_minus, r_minus, theta_plus, r_plus, theta_prime, n_prime, s_prime, alpha_prime, n_alpha_prime = \
                self.build_tree(theta, r, u, v, j-1, epsilon, theta_0, r_0)
            if s_prime == 1:
                if v == -1:
                    theta_minus, r_minus, _, _, theta_2prime, n_2prime, s_2prime, alpha_2prime, n_alpha_2prime = \
                        self.build_tree(theta_minus, r_minus, u, v, j-1, epsilon, theta_0, r_0)
                else:
                    _, _, theta_plus, r_plus, theta_2prime, n_2prime, s_2prime, alpha_2prime, n_alpha_2prime = \
                self.build_tree(theta_plus, r_plus, u, v, j-1, epsilon, theta_0, r_0)
                random_prob = float(np.random.rand(1))
                if (n_prime / max(1, n_prime + n_2prime)) >= random_prob:
                    theta_prime = theta_2prime
                alpha_prime += alpha_2prime
                n_alpha_prime += n_alpha_2prime
                s_prime = s_2prime * self.stop_indicator(theta_minus, r_minus, theta_plus, r_plus)
                n_prime += n_2prime
            return theta_minus, r_minus, theta_plus, r_plus, theta_prime, n_prime, s_prime, alpha_prime, n_alpha_prime
        
    def run_NUTS6(self, M, M_adapt, max_height=10):    
        epsilon = self.find_reasonable_epsilon()
        #print('Initial distributions:')
        #print(self.theta)
        mu = np.log(10 * epsilon)
        epsilon_bar = 1
        H_bar = 0.
        gamma = 0.05
        t_0 = 10
        kappa = 0.75
        self.fitness_hist.append(self.fitness_fn(self.pad(self.theta)))
        for m in range(M):
            print('--------------------------------')
            print(f'Step {m+1} / {M}')
            print(f'epsilon = {epsilon}')
            r_0 = torch.randn_like(self.r)
            #changed 0 to 1e-10 to avoid possible log(0) in build_tree
            u = np.random.uniform(1e-12, self.p_fn(theta=self.theta, r=r_0))
            theta_minus, theta_plus = self.theta, self.theta
            r_minus, r_plus = self.r, self.r
            j, n, s = 0, 1, 1
            while s == 1 and j <= max_height:
                #print(f'Height of the tree = {j}')
                v = np.random.choice([-1, 1])
                if v == -1:
                    theta_minus, r_minus, _, _, theta_prime, n_prime, s_prime, alpha, n_alpha = \
                        self.build_tree(theta_minus, r_minus, u, v, j, epsilon, self.theta, r_0)
                else:
                    _, _, theta_plus, r_plus, theta_prime, n_prime, s_prime, alpha, n_alpha = \
                        self.build_tree(theta_plus, r_plus, u, v, j, epsilon, self.theta, r_0)
                if s_prime == 1:
                    random_prob = float(np.random.rand(1))
                    if min(1, n_prime / n) >= random_prob:
                        self.theta = theta_prime
                n += n_prime
                s = s_prime * self.stop_indicator(theta_minus, r_minus, theta_plus, r_plus)
                j += 1
            if m < M_adapt:
                H_bar = (1 - 1/(m + t_0)) * H_bar + 1/(m + t_0) * (self.delta - alpha / n_alpha)
                epsilon = np.exp(mu - H_bar * np.sqrt(m+1) / gamma)
                #re-wrote: log(epsilon_bar) = m ** (-kappa) * log(epsilon) + ( 1 - m ** (-kappa)) * log(epsilon_bar)
                epsilon_bar = epsilon_bar * (epsilon * epsilon_bar) ** (1 / (m+1) ** kappa)
            #print('Final distributions:')
            #print(theta_prime)
            plt.plot(self.fitness_hist, linestyle="",marker=".")
            plt.title(f'Fitness history after {m+1} iterations')
            plt.show()
                
#--------------------------- EXAMPLE ----------------------------------------
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/Users/castrr/Documents/GitHub/boda2/')    #edit path to boda2
    from boda.common import constants, utils 

    np.set_printoptions(precision=2, suppress=True)    # for shorter display of np arrays
    
    model = NUTS(fitness_fn=utils.first_token_rewarder,
                num_sequences=1,
                seq_len=200,
                padding_len=0,
                upPad_DNA=constants.MPRA_UPSTREAM,
                downPad_DNA=constants.MPRA_DOWNSTREAM,
                vocab_list=constants.STANDARD_NT,
                seed=None,
                temperature=1,
                kinetic_scale_factor=0.01)
       
    #model.run_HMC(num_steps=500, epsilon=0.1)
    model.run_NUTS6(M=20, M_adapt=6)