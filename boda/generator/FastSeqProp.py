import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

class FastSeqProp(nn.Module):
    def __init__(self,
                 energy_fn,
                 params,
                 **kwargs):
        super().__init__()
        self.energy_fn = energy_fn
        self.params = params                           
        self.grad = torch.autograd.grad

        try: self.energy_fn.eval()
        except: pass
    
    def run(self, steps=20, learning_rate=0.5, step_print=5, lr_scheduler=True, create_plot=True):
     
        if lr_scheduler: etaMin = 1e-6
        else: etaMin = learning_rate
        
        optimizer = torch.optim.Adam(self.params.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=etaMin)  
        
        energy_hist = []
        pbar = tqdm(range(1, steps+1), desc='Steps', position=0, leave=True)
        for step in pbar:
            optimizer.zero_grad()
            sampled_nucleotides = self.params()
            energy = self.energy_fn(sampled_nucleotides)
            energy.backward()
            optimizer.step()
            scheduler.step()
            energy_hist.append(energy.item())
            if step % step_print == 0:
                pbar.set_postfix({'Loss': energy.item(), 'LR': scheduler.get_last_lr()[0]})
                
        if create_plot:
            plt.plot(energy_hist)
            plt.xlabel('Steps')
            vert_label=plt.ylabel('Energy')
            vert_label.set_rotation(90)
            plt.show()