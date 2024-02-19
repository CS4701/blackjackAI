import torch
import torch.nn as nn
from torch.nn import functioanl as F
import utili

#hyperparameter 
batch_size = 32 # number of sequences we process in parallel
block_size = 8 #max context length for predictions
max_iter = utili.MAX_ITER
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'gpu'
eval_iters = 200

torch.manual_seed(utili.MANUAL_SEED)

with open('Conversation.csv', 'r', encoding = 'utf-8') as f:
  text = f.read()

#set of unique characters in our dataset file
chars = sorted(list(set(text)))
vocab_size = len(chars)

#mapping of character to integer to create vector-based model
char_to_int = {c: i for i , c in enumerate(chars)}
int_to_char = {i: c for i , c in enumerate(chars)}

decode_text = lambda l: ''.join([int_to_char[i] for i in l]) # decodes a list of integers as a string.  
encode_text = lambda str: [char_to_int[c] for c in str] # encodes a string into a list of integers

#split data into training, validation, and test sets
data = torch.tensor(encode_text(text), dtype=torch.long)

train= int(0.8 * len(data))
val = int(0.9 * len(data))
train_data = data[:train]
val_data = data[train:val]
test_data = data[val:]

#data loading
def get_batch(split):
  if split == 'train':
    data = train_data
  elif split == 'val':
    data = val_data
  else:
    data = test_data 
  #generate a batch of input x
  idx = torch.randint(len(data) - block_size, (batch_size,))
  #generate a batch of target y
  x = torch.stack([data[i : i + block_size] for i in idx])
  y = torch.stack([data[i+1 : i+block_size+1] for i in idx])
  x, y = x.to(device), y.to(device)
  return x, y
          
@torch.no_grad
def approximate_loss():
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split)
      logits, loss = model(X, Y)
      losses[k] - loss.item()
    out[split] = losses.mean()
  model.train()
  return out


class BigramLM(nn.Module):
  def __init__(self, vocab_size):
    super.__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
  
  def forward(self, idx, targets = None):
    # idx and targets are both (B, T) tensor of integers
    logits = self.token_embedding_table(idx) #(B, T, C)

    if targets is None:
      loss = None
    else: 
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss
  
  def generate(self, idx, max_new_tokens):
    #idx is (B,T) array of indices in the current context
    for _ in range(max_new_tokens):
      # get predictions
      logits, loss = self(idx)
      # focus only on the last time step
      logits = logits[:, -1, :] #becomes (B, C)
      #apply softmax to gte probabilities
      probs = F.softmax(logits, dim = 1) #(B, C)
      #sample from the distribution
      idx_next = torch.multinomial(probs, num_samples = 1) #(B, C)
      #append sampled index to the running sequence
      idx - torch.cat((idx, idx_next), dim = 1)
    return idx

model = BigramLM(vocab_size)
m = model.to(device)

#pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

for iter in range(max_iter):
    if iter % eval_iters == 0:
      losses = approximate_loss()
      print(f"step {iter}: train loss {losses['train']:.4f},  val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none= True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype = torch.long, device=device)
print(decode_text(list(m.generate(context, max_new_tokens=500)[0])))


