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
    def __init__(self,
                 num_sequences=1,
                 seq_len=200, 
                 padding_len=0,
                 upPad_DNA=None,
                 downPad_DNA=None,
                 vocab_list=['A','G','T','C'],
                 temperature=1,
                 num_st_samples=1,
                 **kwargs):
        super(ST_parameters, self).__init__()
        self.num_sequences = num_sequences
        self.seq_len = seq_len  
        self.padding_len = padding_len
        self.upPad_DNA = upPad_DNA
        self.downPad_DNA = downPad_DNA
        self.vocab_list = vocab_list
        self.temperature = temperature
        self.num_st_samples = num_st_samples
        
        self.softmax = nn.Softmax(dim=1)
        self.vocab_len = len(vocab_list)       
        upPad_logits, downPad_logits = create_paddingTensors()     #a method or import func? or import the pad tensors?
        self.register_buffer('upPad_logits', upPad_logits)
        self.register_buffer('downPad_logits', downPad_logits)
        
        # initialize theta and r
        self.initialize_theta(one_hot=False)
        self.r = torch.randn_like(self.theta) 
            
    def initialize_theta(self, one_hot=False):
        if one_hot:
            theta = np.zeros((self.num_sequences, self.vocab_len, self.seq_len))
            for seqIdx in range(self.num_sequences):
                for step in range(self.seq_len):
                    random_token = np.random.randint(self.vocab_len)
                    theta[seqIdx, random_token, step] = 1      
            self.theta = nn.Parameter(torch.tensor(theta, dtype=torch.float))  
        else:
            self.theta = nn.Parameter(torch.rand((self.num_sequences, self.vocab_len, self.seq_len)))
    
    def forward(self):
        # Generate a Straight-through estimator sample
        softmax_theta = self.softmax(self.theta / self.temperature)
        probs = Categorical(torch.transpose(softmax_theta, 1, 2))
        idxs = probs.sample((self.num_st_samples, ))
        sample_theta_T = F.one_hot(idxs, num_classes=self.vocab_len)   
        sample_theta = torch.transpose(sample_theta_T, 2, 3)
        sample_theta = sample_theta - softmax_theta.detach() + softmax_theta
        sample_theta = torch.cat([ self.upPad_logits, sample_theta, self.downPad_logits], dim=3)
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
