import numpy as np
import torch
import torch.nn as nn
import random
from tqdm import tqdm

from boda.common import utils, constants

# Adapted from https://github.com/samsinai/FLEXS/blob/master/flexs/baselines/explorers/adalead.py
class AdaLead(nn.Module):
    def __init__(self,
                 fitness_fn = None,
                 measured_sequences = None,
                 sequences_batch_size = 6,
                 model_queries_per_batch = 200,
                 seq_len = 200, 
                 padding_len = 400,
                 vocab = constants.STANDARD_NT,
                 mu = 1,
                 recomb_rate = 0,
                 threshold = 0.05,
                 rho = 0,
                 eval_batch_size = 20,
                 **kwargs):
        
        super(AdaLead, self).__init__()
        self.fitness_fn = fitness_fn
        self.measured_sequences = measured_sequences
        self.sequences_batch_size = sequences_batch_size
        self.model_queries_per_batch = model_queries_per_batch
        self.seq_len = seq_len
        self.padding_len = padding_len
        self.vocab = vocab
        self.mu = mu                                    #number of mutations per sequence.
        self.recomb_rate = recomb_rate
        self.threshold = threshold
        self.rho = rho                                  #number of recombinations per generation
        self.eval_batch_size = eval_batch_size
        self.vocab_len = len(self.vocab)
        self.model_cost = 0
        
        upPad_logits, downPad_logits = utils.create_paddingTensors(num_sequences=1,
                                                                   padding_len=self.padding_len, 
                                                                   for_multi_sampling=False)
        self.register_buffer('upPad_logits', upPad_logits)
        self.register_buffer('downPad_logits', downPad_logits)

        #This tensor is used to get the device of the model in .propose_sequences()
        #(The padding tensors are None when padding_len=0, we can't use them as references)
        #The device is used in .run()
        self.register_buffer('device_reference_tensor', torch.zeros(1))    
        self.dflt_device = self.device_reference_tensor.device
        
    def get_fitness(self, sequence_list):
        self.model_cost += len(sequence_list)
        batch = self.string_list_to_tensor(sequence_list).to(self.dflt_device)
        batch = self.pad(batch)
        fitnesses = self.fitness_fn(batch)
        return fitnesses.squeeze().cpu().numpy()
    
    def string_list_to_tensor(self, sequence_list):
        batch_len = len(sequence_list)
        batch_tensor_size = (batch_len, self.vocab_len, self.seq_len)
        batch_tensor = torch.zeros(batch_tensor_size)
        for idx, seq_str in enumerate(sequence_list):
            tensor = utils.dna2tensor(seq_str)
            batch_tensor[idx, :, :] = tensor
        self.register_buffer('batch_tensor', batch_tensor)
        return batch_tensor
             
    def pad(self, tensor):
        if self.padding_len > 0:
            batch_len = tensor.shape[0]
            upPad_logits, downPad_logits = self.upPad_logits.repeat(batch_len, 1, 1), \
                                           self.downPad_logits.repeat(batch_len, 1, 1) 
            return torch.cat([ upPad_logits, tensor, downPad_logits], dim=-1)
        else:
            return tensor
    
    def start_from_random_sequences(self, num_sequences):
        sequence_list = []
        for seq_idx in range(num_sequences):
            sequence_list.append( ''.join(random.choice(self.vocab) for i in range(self.seq_len)) )
        return sequence_list
    
    @staticmethod
    def generate_random_mutant(sequence: str, mu: float, alphabet: str):
        mutant = []
        for s in sequence:
            if random.random() < mu:
                mutant.append(random.choice(alphabet))
            else:
                mutant.append(s)
        return "".join(mutant)
    
    def recombine_population(self, gen):
        # If only one member of population, can't do any recombining
        if len(gen) == 1:
            return gen
        random.shuffle(gen)
        ret = []
        for i in range(0, len(gen) - 1, 2):
            strA = []
            strB = []
            switch = False
            for ind in range(len(gen[i])):
                if random.random() < self.recomb_rate:
                    switch = not switch
                # putting together recombinants
                if switch:
                    strA.append(gen[i][ind])
                    strB.append(gen[i + 1][ind])
                else:
                    strB.append(gen[i][ind])
                    strA.append(gen[i + 1][ind])
            ret.append("".join(strA))
            ret.append("".join(strB))
        return ret
    
    def propose_sequences(self, initial_sequences): #from_last_checkpoint=False):              
        """Propose top `sequences_batch_size` sequences for evaluation."""
        measured_sequence_set = set(initial_sequences)
        measured_fitnesses = self.get_fitness(initial_sequences)
        
        # Get all sequences within `self.threshold` percentile of the top_fitness
        top_fitness = measured_fitnesses.max()
        top_inds = np.argwhere((measured_fitnesses >= top_fitness * (
                                    1 - np.sign(top_fitness) * self.threshold)))
        top_inds = top_inds.reshape(-1).tolist()
        self.initial_top_fitness = top_fitness
        
        parents = [initial_sequences[i] for i in top_inds]
        parents = np.resize(np.array(parents), self.sequences_batch_size,)
        
        sequences = {}
        previous_model_cost = self.model_cost
        while self.model_cost - previous_model_cost < self.model_queries_per_batch:
            # generate recombinant mutants
            for i in range(self.rho):
                parents = self.recombine_population(parents)

            for i in range(0, len(parents), self.eval_batch_size):
                # Here we do rollouts from each parent (root of rollout tree)
                roots = parents[i : i + self.eval_batch_size]
                root_fitnesses = self.get_fitness(roots)

                nodes = list(enumerate(roots))

                while (len(nodes) > 0
                       and self.model_cost - previous_model_cost + self.eval_batch_size
                       < self.model_queries_per_batch):
                    child_idxs = []
                    children = []
                    while len(children) < len(nodes):
                        idx, node = nodes[len(children) - 1]

                        child = self.generate_random_mutant(node,
                                                            self.mu * 1 / len(node),
                                                            self.vocab,)

                        # Stop when we generate new child that has never been seen
                        # before
                        if (child not in measured_sequence_set and child not in sequences):
                            child_idxs.append(idx)
                            children.append(child)

                    # Stop the rollout once the child has worse predicted
                    # fitness than the root of the rollout tree.
                    # Otherwise, set node = child and add child to the list
                    # of sequences to propose.
                    fitnesses = self.get_fitness(children).reshape(-1).tolist()
                    sequences.update(zip(children, fitnesses))

                    nodes = []
                    for idx, child, fitness in zip(child_idxs, children, fitnesses):
                        if fitness >= root_fitnesses[idx]:
                            nodes.append((idx, child))
        if len(sequences) == 0:
            raise ValueError(
                "No sequences generated. If `model_queries_per_batch` is small, try "
                "making `eval_batch_size` smaller")

        # We propose the top `self.sequences_batch_size` new sequences we have generated
        new_seqs = np.array(list(sequences.keys()))
        preds = np.array(list(sequences.values()))
        sorted_order = np.argsort(preds)[: -self.sequences_batch_size : -1]
        return new_seqs[sorted_order], preds[sorted_order]
        
    def run(self, num_iterations=10):
        self.dflt_device = self.device_reference_tensor.device    
        if self.measured_sequences is None:
            self.new_seqs = self.start_from_random_sequences(self.sequences_batch_size)
        else:
            self.new_seqs = self.measured_sequences
        pbar = tqdm(range(num_iterations), desc='Iterations')
        for iteration in pbar:
            self.new_seqs, self.preds = self.propose_sequences(self.new_seqs)
            self.final_top_fitness = max(self.preds)
            pbar.set_postfix({'Initial top fitness': self.initial_top_fitness, 'Final top fitness': self.final_top_fitness})
    