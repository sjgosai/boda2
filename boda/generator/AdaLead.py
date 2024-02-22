import numpy as np
import torch
import torch.nn as nn
import random
from tqdm import tqdm

from boda.common import utils, constants


class AdaLead(nn.Module):
    
    def __init__(self,
                 energy,
                 params,
                 ):

        super().__init__()

        self.energy_fn = energy
        self.params = params
        self.model_cost = 0     
        self.vocab = boda.common.constants.STANDARD_NT

        self.batch_size, self.num_classes, self.seq_len = self.params.theta.shape
        self.token_dim = self.params.token_dim
        self.batch_dim = self.params.batch_dim

        self.dflt_device = self.params.theta.device                  
        try: self.energy_fn.eval()
        except: pass

    @staticmethod
    def add_generator_specific_args(parent_parser):
        """
        Static method to add generator-specific arguments to a parser.

        Args:
            parent_parser (ArgumentParser): Parent argument parser.

        Returns:
            ArgumentParser: Argument parser with added generator-specific arguments.
        """
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        group  = parser.add_argument_group('Generator Constructor args')
        # Empty
        
        group  = parser.add_argument_group('Generator Runtime args')
        group.add_argument('--n_steps', type=int, default=20)
        group.add_argument('--n_top_seqs_per_batch', type=int)
        group.add_argument('--mu', type=float, default=1)
        group.add_argument('--recomb_rate', type=float, default=0.1)
        group.add_argument('--threshold', type=float, default=0.05)
        group.add_argument('--rho', type=int, default=2)
        group.add_argument('--model_queries_per_batch', type=int)
                          
        return parser

    @staticmethod
    def process_args(grouped_args):
        """
        Static method to process grouped arguments.

        Args:
            grouped_args (dict): Dictionary containing grouped arguments.

        Returns:
            tuple: A tuple containing constructor args and runtime args.
        """
        constructor_args = grouped_args['Generator Constructor args']
        runtime_args     = grouped_args['Generator Runtime args']
        
        return constructor_args, runtime_args


    def get_fitness(self, seq_str_list):
        self.model_cost += len(seq_str_list)
        input_tensor = self.string_list_to_tensor(seq_str_list)
        input_tensor = self.params(input_tensor.to(self.dflt_device))
        return -1 * self.energy_fn(input_tensor).detach().cpu().numpy()

    def string_list_to_tensor(self, sequence_list):
        return torch.stack([boda.common.utils.dna2tensor(sequence) for sequence in sequence_list]).to(self.dflt_device)

    def random_string_list(self):
        return [''.join(random.choices(self.vocab, k=self.seq_len)) for _ in range(self.batch_size)]

    def generate_random_mutant(self, sequence, mu_rate):
        return ''.join([random.choice(self.vocab) if random.random() < mu_rate else s for s in sequence])

    def start_from_random_sequences(self, num_sequences):
        return [''.join(random.choices(self.vocab, k=self.seq_len)) for _ in range(num_sequences)]
  
    def recombine_population(self, gen, recomb_rate=0.1):
        if len(gen) == 1:
            return gen
        random.shuffle(gen)
        ret = []
        for i in range(0, len(gen) - 1, 2):
            strA = []
            strB = []
            switch = False
            for ind in range(len(gen[i])):
                if random.random() < recomb_rate:
                    switch = not switch
                if switch:
                    strA.append(gen[i][ind])
                    strB.append(gen[i + 1][ind])
                else:
                    strB.append(gen[i][ind])
                    strA.append(gen[i + 1][ind])
            ret.append("".join(strA))
            ret.append("".join(strB))
        return ret

    def propose_sequences(self, initial_sequences, mu=1, recomb_rate=0.1, threshold=0.05,
                          rho=2, model_queries_per_batch=5000):          
        measured_sequence_set = set(initial_sequences)
        measured_fitnesses = self.get_fitness(initial_sequences)
        
        top_fitness = measured_fitnesses.max()
        top_inds = np.argwhere((measured_fitnesses >= top_fitness * (
                                    1 - np.sign(top_fitness) * threshold)))
        top_inds = top_inds.reshape(-1).tolist()
        
        parents = [initial_sequences[i] for i in top_inds]
        
        sequences = {}
        
        roots = np.resize(np.array(parents), self.batch_size,)

        self.model_cost = 0 
        while self.model_cost < model_queries_per_batch:
            for i in range(rho):
                roots = self.recombine_population(roots, recomb_rate)
            root_fitnesses = self.get_fitness(roots)

            nodes = list(enumerate(roots))

            while (len(nodes) > 0
                    and self.model_cost + self.batch_size
                    < model_queries_per_batch):
                child_idxs = []
                children = []
                while len(children) < len(nodes):  
                    idx, node = nodes[len(children)]  

                    child = self.generate_random_mutant(node, mu / len(node))
                    if (child not in measured_sequence_set and child not in sequences):
                        child_idxs.append(idx)
                        children.append(child)
                fitnesses = self.get_fitness(children).reshape(-1).tolist()
                sequences.update(zip(children, fitnesses))

                nodes = []
                for idx, child, fitness in zip(child_idxs, children, fitnesses):
                    if fitness > root_fitnesses[idx]:
                        nodes.append((idx, child))

        if len(sequences) == 0:
            raise ValueError(
                "No sequences generated. If `model_queries_per_batch` is small, try "
                "making `batch_size` smaller")

        new_seqs = np.array(list(sequences.keys()))
        new_fitnesses = np.array(list(sequences.values()))
        sorted_order = np.argsort(-new_fitnesses)[:self.batch_size]

        return new_seqs[sorted_order], new_fitnesses[sorted_order]


    def run(self, pre_provided_sequences=None, num_iterations=30, mu=1, recomb_rate=0.1, threshold=0.05,
                          rho=2, model_queries_per_batch=5000, desc_str=''):
        if pre_provided_sequences is None:
            new_seqs = self.start_from_random_sequences(self.batch_size)
        else:
            new_seqs = pre_provided_sequences
        new_seqs = self.start_from_random_sequences(self.batch_size)
        pbar = tqdm(range(num_iterations), desc=desc_str, position=0, leave=True)
        for iteration in pbar:
            new_seqs, new_fitnesses = self.propose_sequences(new_seqs, mu=mu, recomb_rate=recomb_rate,
                                                     threshold=threshold, rho=rho,
                                                     model_queries_per_batch=model_queries_per_batch)
            mean_fitness = np.mean(new_fitnesses)
            pbar.set_postfix({'Batch mean fitness': mean_fitness})

        return new_seqs, new_fitnesses


    def generate(self, n_proposals=1, energy_threshold=float("Inf"), n_steps=20, n_top_seqs_per_batch=None,
                 mu=1, recomb_rate=0.1, threshold=0.1, rho=2, model_queries_per_batch=None, max_attempts=10000):
        
        if n_top_seqs_per_batch is None:
            n_top_seqs_per_batch = self.batch_size
        else:
            n_top_seqs_per_batch = min(n_top_seqs_per_batch, self.batch_size)

        if model_queries_per_batch is None:
            model_queries_per_batch = self.batch_size * 10

        print(f'Collecting at most {n_top_seqs_per_batch} sequences per batch', flush=True)
        
        proposals = []
        energies  = []
        acceptance = []
        batch_idx = 1
        attempts = 0
        print('', flush=True)
        while (len(proposals) < n_proposals) and (attempts <= max_attempts):
            desc_str = f'Batch {batch_idx} ({len(proposals)}/{n_proposals} proposals generated )'
            batch_proposals, batch_fitnesses = self.run(mu=mu, recomb_rate=recomb_rate, num_iterations=n_steps,
                                                        threshold=threshold, rho=rho, desc_str=desc_str,
                                                        model_queries_per_batch=model_queries_per_batch)
            attempts += len(batch_proposals)
            passing_idxs = np.where(-batch_fitnesses <= energy_threshold)[0]
            passing_proposals = np.array(batch_proposals)[passing_idxs].tolist()
            passing_energies = -batch_fitnesses[passing_idxs]
            proposals.extend(passing_proposals[:n_top_seqs_per_batch])
            energies.extend(passing_energies[:n_top_seqs_per_batch].tolist())
            acceptance.append(len(passing_idxs) / self.batch_size)
            batch_idx += 1
        print()
        
        proposals = torch.stack([boda.common.utils.dna2tensor(proposal) for proposal in proposals[:n_proposals]])
        energies = torch.Tensor(energies[:n_proposals])
        acceptance = np.mean(acceptance)

        print(f'{proposals.shape[0]} proposals generated')

        results = {
            'proposals': proposals,
            'energies': energies,
            'acceptance_rate': acceptance
        }
        
        return results
    
    

class deprecated_AdaLead(nn.Module):
    """
    Adapt-with-the-Leader (AdaLead) module for sequence optimization.
    Adapted from https://github.com/samsinai/FLEXS/blob/master/flexs/baselines/explorers/adalead.py

    Args:
        fitness_fn (callable): A function to evaluate the fitness of sequences.
        measured_sequences (list[str]): List of initial sequences to start optimization from.
        sequences_batch_size (int): Number of sequences in each batch during optimization.
        model_queries_per_batch (int): Maximum number of fitness evaluations per batch.
        seq_len (int): Length of each sequence.
        padding_len (int): Length of padding added to each sequence for model queries.
        vocab (str): The vocabulary of characters for sequences.
        mu (float): Number of mutations per sequence during mutation step.
        recomb_rate (float): Recombination rate for generating new sequences.
        threshold (float): Threshold percentile for selecting top-performing sequences.
        rho (int): Number of recombinations per generation.
        eval_batch_size (int): Number of sequences to evaluate fitness concurrently.
        **kwargs: Additional keyword arguments.

    Attributes:
        fitness_fn (callable): The fitness evaluation function.
        measured_sequences (list[str]): Initial sequences for optimization.
        sequences_batch_size (int): Batch size of sequences during optimization.
        model_queries_per_batch (int): Maximum model queries per batch.
        seq_len (int): Length of sequences.
        padding_len (int): Length of padding added to sequences.
        vocab (str): Vocabulary of characters for sequences.
        mu (float): Number of mutations per sequence.
        recomb_rate (float): Recombination rate for generating new sequences.
        threshold (float): Threshold percentile for selecting top-performing sequences.
        rho (int): Number of recombinations per generation.
        eval_batch_size (int): Number of sequences to evaluate fitness concurrently.
        vocab_len (int): Length of the vocabulary.
        model_cost (int): Total cost of fitness evaluations.
        upPad_logits (Tensor): Tensor for padding sequences upstream.
        downPad_logits (Tensor): Tensor for padding sequences downstream.
        device_reference_tensor (Tensor): Reference tensor for device information.
        dflt_device (device): Default device for computations.

    Methods:
        get_fitness(sequence_list): Evaluate fitness of a list of sequences.
        string_list_to_tensor(sequence_list): Convert list of sequences to tensor.
        pad(tensor): Pad sequences with padding tensors.
        start_from_random_sequences(num_sequences): Generate random initial sequences.
        generate_random_mutant(sequence, mu, alphabet): Generate a random mutant sequence.
        recombine_population(gen): Recombine sequences in a population.
        propose_sequences(initial_sequences): Propose new sequences for optimization.
        run(num_iterations, desc_str): Run the AdaLead optimization process.

    Note:
        - This class is designed for sequence optimization using the AdaLead algorithm.
        - During optimization, it proposes new sequences based on fitness evaluations.

    """
    
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
        """
        Initialize the AdaLead optimizer.

        Args:
            fitness_fn (callable): A function to evaluate the fitness of sequences.
            measured_sequences (list[str]): List of initial sequences to start optimization from.
            sequences_batch_size (int): Number of sequences in each batch during optimization.
            model_queries_per_batch (int): Maximum number of fitness evaluations per batch.
            seq_len (int): Length of each sequence.
            padding_len (int): Length of padding added to each sequence for model queries.
            vocab (str): The vocabulary of characters for sequences.
            mu (float): Number of mutations per sequence during mutation step.
            recomb_rate (float): Recombination rate for generating new sequences.
            threshold (float): Threshold percentile for selecting top-performing sequences.
            rho (int): Number of recombinations per generation.
            eval_batch_size (int): Number of sequences to evaluate fitness concurrently.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
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
        #since the padding tensors are None is padding_len=0.
        #The device is used in .run()
        self.register_buffer('device_reference_tensor', torch.zeros(1))    
        self.dflt_device = self.device_reference_tensor.device
        
    def get_fitness(self, sequence_list):
        """
        Evaluate the fitness of a list of sequences using the fitness function.

        Args:
            sequence_list (list[str]): List of sequences to evaluate.

        Returns:
            ndarray: Array of fitness values for the input sequences.
        """
        self.model_cost += len(sequence_list)
        batch = self.string_list_to_tensor(sequence_list).to(self.dflt_device)
        batch = self.pad(batch)
        fitnesses = self.fitness_fn(batch)
        return fitnesses.squeeze().cpu().detach().numpy()
    
    def string_list_to_tensor(self, sequence_list):
        """
        Convert a list of sequences to a tensor representation.

        Args:
            sequence_list (list[str]): List of sequences.

        Returns:
            Tensor: Tensor containing the one-hot-encoded sequences.
        """
        return torch.stack([utils.dna2tensor(sequence) for sequence in sequence_list])
             
    def pad(self, tensor):
        """
        Pad sequences with padding tensors.

        Args:
            tensor (Tensor): Input tensor of sequences.

        Returns:
            Tensor: Padded tensor containing sequences with padding.
        """
        if self.padding_len > 0:
            batch_len = tensor.shape[0]
            upPad_logits, downPad_logits = self.upPad_logits.repeat(batch_len, 1, 1), \
                                           self.downPad_logits.repeat(batch_len, 1, 1) 
            return torch.cat([ upPad_logits, tensor, downPad_logits], dim=-1)
        else:
            return tensor
    
    def start_from_random_sequences(self, num_sequences):
        """
        Generate random initial sequences.

        Args:
            num_sequences (int): Number of random sequences to generate.

        Returns:
            list[str]: List of randomly generated sequences.
        """
        sequence_list = []
        for seq_idx in range(num_sequences):
            sequence_list.append( ''.join(random.choice(self.vocab) for i in range(self.seq_len)) )
        return sequence_list
    
    @staticmethod
    def generate_random_mutant(sequence: str, mu: float, alphabet: str):
        """
        Generate a random mutant sequence based on the given sequence.

        Args:
            sequence (str): Input sequence to mutate.
            mu (float): Mutation rate for generating the mutant.
            alphabet (str): Alphabet of characters for the sequence.

        Returns:
            str: Mutated sequence.
        """
        mutant = []
        for s in sequence:
            if random.random() < mu:
                mutant.append(random.choice(alphabet))
            else:
                mutant.append(s)
        return "".join(mutant)
    
    def recombine_population(self, gen):
        """
        Recombine sequences in a population using crossover.

        Args:
            gen (list[str]): List of sequences in the population.

        Returns:
            list[str]: List of recombinant sequences.
        """
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
        """
        Propose top sequences_batch_size sequences for evaluation.

        Args:
            initial_sequences (list[str]): Initial sequences to start optimization from.

        Returns:
            tuple: A tuple containing new sequences and their predicted fitness values.
        """         
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
                        idx, node = nodes[len(children)]

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
                        if fitness > root_fitnesses[idx]:
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

    def run(self, num_iterations=10, desc_str='Iterations'):
        """
        Run the AdaLead optimization process for a specified number of iterations.

        Args:
            num_iterations (int): Number of iterations to run.
            desc_str (str): Description string for the progress bar.

        Returns:
            None
        """
        self.dflt_device = self.device_reference_tensor.device    
        if self.measured_sequences is None:
            new_seqs = self.start_from_random_sequences(self.sequences_batch_size)
            #print('Starting from random sequences')
        else:
            new_seqs = self.measured_sequences
            #print('Starting from given initial sequences')
        pbar = tqdm(range(num_iterations), desc=desc_str, position=0, leave=True)
        for iteration in pbar:
            new_seqs, preds = self.propose_sequences(new_seqs)
            final_top_fitness = max(preds)
            pbar.set_postfix({'Initial top fitness': self.initial_top_fitness, 'Final top fitness': final_top_fitness})
        self.new_seqs = new_seqs
        self.preds = preds
        
        
    