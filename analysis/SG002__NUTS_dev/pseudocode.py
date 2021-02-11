import torch.autograd as ag

class NUTS(nn.Module):
  def __init__(self, fitness_callable, step_size=1., adapt_step_size=False, target_accept_prob=0.8, max_tree_depth=10):
    return None

class SMT_parameters(nn.Module):
  def __init__(self, size=(1,4,600), temperature):
    self.theta = nn.Parameter(torch.randn(size=size))
    self.sm    = nn.Softmax(dim=1)
    self.temperature = temperature
  def forward(self):
    return self.sm( self.theta / self.temperature )
  
  
class ST_parameters(nn.Module):
  def __init__(self, num_sequences=1, vocab_len=4, seq_len=200, temperature=1, upPad_tensor, downPad_tensor):
    self.num_sequences = num_sequences
    self.vocab_len = vocab_len
    self.seq_len = seq_len
    self.upPad_tensor = upPad_tensor
    self.temperature = temperature
    self.downPad_tensor = downPad_tensor
    self.softmax = nn.Softmax(dim=1)
    self.theta = nn.Parameter( torch.rand(self.num_sequences, self.vocab_len, self.seq_len) )
    self.register_buffer('upPad_tensor', upPad_tensor)
    self.register_buffer('downPad_tensor', downPad_tensor)
    
  def forward(self):
    # Generate a Straight-through estimator sample
    softmax_theta = self.softmax(self.theta / self.temperature)
    probs = Categorical(torch.transpose(softmax_theta, 1, 2))
    idxs = probs.sample()
    sample_theta_T = F.one_hot(idxs, num_classes=self.vocab_len)   
    sample_theta = torch.transpose(sample_theta_T, 1, 2)
    sample_theta = sample_theta - softmax_theta.detach() + softmax_theta
    sample_theta = torch.cat([ self.upPad_logits, sample_theta, self.downPad_logits], dim=2)
    return sample_theta
  

class NUTS3(nn.Module):
  def __init__(self, fitness_fn, theta, epsilon):
    self.fitness_fn = fitness_fn
    self.theta      = theta
    self.epsilon    = epsilon
    self.r          = torch.randn_like(self.theta)
    
    self.fitness_fn( self.theta ).sum().backward()
    
  def leapfrog(self, r, grad_):
    
    r.data -= grad_.mul(self.epsilon).div(2.)
    
    with torch.no_grad():
      self.theta.data += r.mul(self.epsilon)
      
    grad_.data = ag.grad( self.fitness_fn(self.theta).sum(), self.theta )[0]
    
    r.data -= grad_.mul(self.epsilon).div(2.)
    
    return r, grad_
  
  def leapfrog(self):
    
    self.r.data -= self.theta.data.grad * self.epsilon / 2.
    
    self.theta.zero_grad()
    self.theta.data += r * self.epsilon
    self.fitness_fn( self.theta ).sum().backward()
    
    self.r.data -= self.theta.data.grad * self.epsilon / 2.
    
  
  
  
model  = Basset(...)
model.eval()
params = SMT_parameters(...)

a_sample = params()
a_fitness= model.fitness_fn(a_sample)


generator = NUTS3(model.fitness_fn, params, epsilon=1.0)
my_samples  = generator.fit(samples=1000, burnin=1000)
# where:
# my_samples := (seq_batches, potential_energy_readings)
