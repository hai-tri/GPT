import torch
import torch.nn as nn
import time
from torch.nn import functional as F
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import urllib.request
import matplotlib.pyplot as plt
import os



batch_size = 128 
context = 512
max_iters = 5000
eval_interval = 250
learning_rate = 1e-4
device = 'cuda' 
eval_iters = 200
n_embd = 768
n_head = 12
n_layer = 24
dropout = 0.1
numgpu = 2


if (int(os.environ.get('RANK', -1)) != -1): # grabs the 'rank' variable if it exists and returns its value as a string. if else, returns default val of -1.
    
    assert torch.cuda.is_available(), "Remember to connect to cluster!"
    init_process_group(backend='nccl')
    rank = int(os.environ['RANK']) #refers to the id of a singular node (gpu, ect.)
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK']) # if running on several nodes (several boxes of 8 gpus)
    #due to having several nodes, it would be helpful to include their respective names while running
    device = f'cuda:{local_rank}'
    torch.cuda.set_device(device)
    head_process = rank == 0 #checks whether the gpu is the "head node" while the others perform primarily backprop/forward prop
else:
    rank = 0
    world_size = 1
    local_rank = 0
    head_process = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")

def cleanup():
    dist.destroy_process_group()

# print(f"batch: {batch_size} | context: {context} | D_embed: {n_embd} | dropout: {dropout}")

torch.cuda.manual_seed(9876) if torch.cuda.is_available() else torch.manual_seed(9876)# cuda manual seed affects current gpu

if head_process:
    url = "https://raw.githubusercontent.com/nattrng/GPT/main/orwell.txt"
    urllib.request.urlretrieve(url, 'input.txt')

if (int(os.environ.get('RANK', -1)) != -1):
    dist.barrier()

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] 
decode = lambda l: ''.join([itos[i] for i in l]) 
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) 
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - context, (batch_size,))
    x = torch.stack([data[i:i+context] for i in ix])
    y = torch.stack([data[i+1:i+context+1] for i in ix])

    x, y = x.to(device), y.to(device)
    return x, y

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

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context, context)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x) 
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd) 
        self.dropout = nn.Dropout(dropout) 

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    

class DynamicTanh(nn.Module):

    def __init__(self, n_embed):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(n_embed))
        self.beta = nn.Parameter(torch.zeros(n_embed))
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return self.gamma + torch.nn.functional.tanh(self.alpha * x) + self.beta
    

    
class affine_1n(nn.Module):
    """test affine"""

    def __init__(self, n_embd, n_latent):
        super().__init__()
        self.affine = nn.Linear(n_embd, n_latent, bias=True)

    def forward(self, x):
        return self.affine(x)

# class SparseAutoEncoder(nn.Module):

#     """test sparse autoencoder by myself"""

#     def __init__(self, n_embed, n_latent, n_affines):
#         super().__init__()
#         self.affinetransform = nn.Sequential(*[affine_1n(n_embed, n_latent) for _ in range(n_affines)])
        
#     def forward(self, x):
#         return self.affinetransform(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.dynt = DynamicTanh(n_embd)
        self.dynt2 = DynamicTanh(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x)) 
        return x
    

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(context, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) 
        self.lm_head = nn.Linear(n_embd, vocab_size)
        # self.SparseAutoEncoder = SparseAutoEncoder(n_embd, 16)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) 
        x = tok_emb + pos_emb 
        x = self.blocks(x)
        x = self.ln_f(x) 
        logits = self.lm_head(x) 

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) 
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BigramLanguageModel()
m = model.to(device)

if (int(os.environ.get('RANK', -1)) != -1):
    model = DDP(model, device_ids=[local_rank])
    raw_model = model.module # unwraps the model from DDP
else:
    raw_model = model

if head_process:
    print(sum(p.numel() for p in raw_model.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iters)

start_time = time.time()

lossi = []
stepi = []
epoch_count = 0
approx_epoch = 0

for iter in range(max_iters):

    xb, yb = get_batch('train')

    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        if head_process:
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} | Truthful Epoch(s): {epoch_count} | Approximate Epoch(s): {approx_epoch}")

    # evaluate the loss
    logits, loss = model(xb, yb)
    if head_process:
        lossi.append(loss.cpu().item())
        stepi.append(iter)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    scheduler.step()

end_time = time.time()
elapsed_time = end_time - start_time
if head_process:
    context_tensor = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(raw_model.generate(context_tensor, max_new_tokens=2000)[0].tolist()))
    print(f"\n\nElapsed training time ({device}): {elapsed_time:.2f} seconds")


    plt.plot(stepi, lossi)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.show()

if (int(os.environ.get('RANK', -1)) != -1):
    destroy_process_group()


# Test Config for 2 DDP H100 Usage
# batch_size = 256 
# context = 512
# max_iters = 5000
# eval_interval = 250
# learning_rate = 1e-4
# device = 'cuda' 
# eval_iters = 200
# n_embd = 768
# n_head = 12
# n_layer = 24
# dropout = 0.1
# numgpu = 2