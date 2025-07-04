---
layout: post-wide
title:  "Building GPT from Scratch - Code Explanation"
date:  2024-10-17 22:41:32 +0800
category: AI 
author: Hank Li
---
Andrej Karpathy's "Let's build GPT: from scratch, in code, spelled out" tutorial ([code](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing#scrollTo=hoelkOrFY8bN), [video](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=3483s)) offers an invaluable hands-on approach to demystifying one of the most influential architectures in modern AI. This blog post walks through the key concepts and implementations from Karpathy's tutorial, providing a clear path to understanding the inner workings of Generative Pre-trained Transformers (GPT).

This tutorial breaks down the complexity into digestible components, showing how each piece contributes to the model's remarkable ability to generate human-like text. By building a GPT model from the ground up, we gain insights that would be impossible to achieve through theoretical study alone.

Unlike high-level explanations, this tutorial dives into the actual code implementation, showing how concepts like positional encoding, layer normalization, and feed-forward networks are implemented in practice. You'll see how these components work together to create a functioning language model.

By the end of the tutorial, you'll have a working GPT model that can generate text, complete prompts, and demonstrate many of the capabilities we associate with large language models. This hands-on experience bridges the gap between theory and practical application.

## 1. Setup and Hyperparameters

```python
import torch
import torch.nn as nn
from torch.nn import functional as F

### Hyperparameters

# how many independent sequences will we process in parallel? 
# GPT2 uses 512 sequences, estimated batch size for GPT-3 is 1,000 - 4,000 sequences
batch_size = 16 

# what is the maximum context length for predictions? 
# ChatGPT-3: 2048 (GPT-3's context window)
block_size = 32 

max_iters = 5000 # total training iterations
eval_interval = 100 # how often to evaluate loss
learning_rate = 1e-3 # step size for optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu' # use GPU if available
eval_iters = 200 # number of iterations to estimate loss

# embedding dimension (size of token representations) 
# GPT3 170B model embedding dimension is 12,288.
n_embd = 64 
n_head = 4 # number of attention heads; ChatGPT-3: 96 attention heads
n_layer = 4 # number of transformer layers; 96 layers in ChatGPT-3 model
dropout = 0.0 # regularization to prevent overfitting. set to 0 for simplicity. GPT-3, the dropout rate is 0.1
```

## 2. Vocabulary Processing

```python
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# ChatGPT-3 uses Byte Pair Encoding (BPE):
# - Merges frequent character sequences into tokens
# - Balances vocabulary size vs sequence length
# - Handles rare words better than pure char-level
# - Typical vocab size ~50k tokens

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
```
Why Subword Tokens?
 - More efficient than chars (shorter sequences)
 - Better than word-level (handles unknown words)
 - "Un" + "happy" = "Unhappy" (morphological understanding)


## 3. Data Loading 
 ```python
# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    #random chunks from single document
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
 ```
ChatGPT-3 loading:
1. Trained on 300B tokens from CommonCrawl, books, Wikipedia
2. Uses sophisticated preprocessing (quality filtering, deduplication)
3. Batches are carefully constructed to maintain document boundaries
4. Uses much larger context windows/block size (2048 tokens)

## 4. Attention Mechanism Deep Dive

```python
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        # the current example n_embd: 64 --> head_size: 16
        # ChatGPT-3: (batch, 2048, 128) per head (12288/96=128): n_embd: 12288 ---> head_size: 128
        self.key = nn.Linear(n_embd, head_size, bias=False) # 64->16 
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # register_buffer vs nn.Parameter:
        # - Buffer: Non-trainable (like mask)
        # - Parameter: Trainable (like weights)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        # - Normally drops some attention scores (but 0 here)
        # - ChatGPT-3 uses dropout=0.1 during training
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape

        # For n_embd=64, n_head=4:
        # - Each head gets 16 dimensions (64/4)
        # - Key/Query/Value projections reduce to head_size
        # - Final concatenation restores original dimension; see MultiHeadAttention
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)

        # - Creates causal mask for variable length inputs
        # - Prevents lookahead beyond current sequence length
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)

        wei = F.softmax(wei, dim=-1) # (B, T, T)

        # Dropout in attention (wei = self.dropout(wei)):
        # - Helps prevent over-reliance on specific attention patterns
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out
```

## 5. Multi-Head Attention Projection

```python
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()

        # Heads compute attention independentl
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # - Final concatenation restores original dimension
        # 1. Recombines information from all heads
        out = torch.cat([h(x) for h in self.heads], dim=-1)

        # 2. Allows heads to interact before next layer
        # 3. Projects back to standard dimension
        out = self.dropout(self.proj(out))
        return out`
```

## 6. Feedforward Network Role
```python
class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            # ChatGPT-3 uses similar structure but with:
            # - GELU activation instead of ReLU
            # - Larger expansion factor (4x is standard)
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
```
Why FFN Matters:
- Processes attended information non-linearly
- Acts as "memory" storing factual knowledge
- Research shows most factual recall happens in FFN layers

## 7. Transformer Block Design
```python
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Original paper found this pattern works best
        # one MultiHeadAttention followed by one Feedforward
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
```

## 8. Super Simple Bigram Model
```python
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # ChatGPT-3 uses sinusoidal positional encodings:
        # 1. Fixed (not learned) patterns
        # 2. Alternating sine/cosine waves
        # 3. Allows extrapolation beyond training length
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # Why Multiple Layers?
        # Lower layers: basic syntax/local patterns
        # Middle layers: grammatical structures
        # Higher layers: long-range dependencies/semantics
        # More layers = more abstraction (but diminishing returns)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])

        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
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
```

## 9. Train the model
```python

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model = BigramLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))
```
