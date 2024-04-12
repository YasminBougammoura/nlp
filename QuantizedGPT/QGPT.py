import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
import time
import csv
import matplotlib.pyplot as plt

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 2000
eval_interval = 200
learning_rate = 1e-3
eval_iters = 200
n_embd = 32
eps = 1e-5
n_head = 4
n_layer = 6
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

loss_val = []
loss_train = []

with open('\input.txt','r', encoding = 'utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    qmodel.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = qmodel(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    qmodel.train()
    return out

class BitLinear(nn.Module):

  def __init__(self, in_features, out_features, bits):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.bits = bits
    self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
    self.bias = nn.Parameter(torch.Tensor(out_features))

    self.reset_parameters()

  def reset_parameters(self):

    stdv = 1.0 / math.sqrt(self.weight.size(1))
    self.weight.data.uniform_(-stdv, stdv)
    self.bias.data.uniform_(-stdv, stdv)

  def quantization(self, input_data):

    Qb = 2**(self.bits - 1)

    abs_mean = torch.mean(torch.abs(self.weight)) + eps
    W = self.weight/abs_mean
    adj_par = 2**(self.bits-1)/torch.max(torch.abs(input_data) + eps)
    X = adj_par * input_data

    if self.training:
      Wq = W + (torch.clip(torch.round(W),-1,1) - W).detach()
      Xq = X + (torch.clip(torch.round(X), -Qb + eps, Qb - eps) - X).detach()
    else:
      Wq = torch.clip(torch.round(W),-1,1)
      Xq = torch.clip(torch.round(X), -Qb + eps, Qb - eps)

    Xq = torch.clip(torch.round(Xq), -2**(self.bits-1)+eps, 2**(self.bits-1)-eps)

    return Xq,Wq

  def forward(self,x):
    ln = nn.LayerNorm(x.shape[2]).to(device)
    X_q,W_q = self.quantization(ln(x))

    gamma = torch.max(torch.abs(x))
    beta = torch.mean(torch.abs(self.weight))
    deq_par = (beta*gamma)/(2**(self.bits-1))

    y = F.linear(X_q, W_q, self.bias)
    y = y * deq_par

    return y
  
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = BitLinear(n_embd, head_size,8)
        self.query = BitLinear(n_embd, head_size,8)
        self.value = BitLinear(n_embd, head_size,8)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * self.head_size **-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)

        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out
    
class MultiHeadAttention(nn.Module):

  def __init__(self,num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = BitLinear(n_embd, n_embd,8)
    self.dropout = nn.Dropout(dropout)

  def forward(self,x):
    out = torch.cat([h(x) for h in self.heads], dim = -1) #concat over channel dim
    out = self.proj(out)
    return out
  
class FeedForward(nn.Module):

  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
        BitLinear(n_embd,4*n_embd,8), # *4 secondo argomento
        nn.ReLU(),
        BitLinear(4*n_embd, n_embd,8), #proj layer #*4 primo argomento
        nn.Dropout(dropout),
    )

  def forward(self,x):
    return self.net(x)
  
class Block(nn.Module):

  def __init__(self, n_embd, n_head):
    super().__init__()
    head_size = n_embd//n_head
    self.sa = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedForward(n_embd)

  def forward(self,x):
    x = x + self.sa(x)
    x = x + self.ffwd(x)
    return x
  
# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        #self.bitlayer = BitLinear(n_embd,vocab_size,8)
        self.linear = nn.Linear(n_embd,vocab_size)


    def forward(self, idx, targets=None):
        B,T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device = device)) #(T,C)
        x = tok_emb + pos_emb #(B,T,C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.linear(x) #self.bitlayer(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.reshape(B*T, C)
            targets = targets.reshape(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):

            idx_cond = idx[:,-block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

qmodel = BigramLanguageModel()
q = qmodel.to(device)

print(sum(p.numel() for p in q.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(qmodel.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        loss_val.append(losses['val'])
        loss_train.append(losses['train'])
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = qmodel(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

plt.plot(np.arange(0,max_iters,eval_interval), loss_val, label = f'Validation loss')
plt.plot(np.arange(0,max_iters,eval_interval), loss_train, label = f'Training loss')

plt.xlabel('Number of iterations')
plt.ylabel('Loss')
plt.title('Weight and Input Quantization')
plt.legend()
plt.grid(True)
plt.savefig('LossQGPT.png')

#context = torch.zeros((1, 1), dtype=torch.long, device=device)
#print(decode(q.generate(context, max_new_tokens=500)[0].tolist()))
