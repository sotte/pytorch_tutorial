# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # RNN from scratch with PyTorch
# A RNN ist just a normal NN.
# It's very easy to implement in PyTorch due to its dynamic nature.
#
# We'll build a very simple character based language model.
#
# Taken from http://www.fast.ai/

# %% [markdown]
# ## Init and helpers

# %%
from pathlib import Path
import numpy as np

# %% [markdown]
# ## Data

# %%
NIETSCHE_PATH = Path("../data/raw/nietzsche.txt")
if NIETSCHE_PATH.is_file():
    print("I already have the data.")
else:
    # !wget -o ../data/raw/nietzsche.txt https://s3.amazonaws.com/text-datasets/nietzsche.txt
        
with NIETSCHE_PATH.open() as f:
    data = f.read()

# %% [markdown]
# A tweet of Nietzsche:

# %%
print(data[:140])

# %% [markdown]
# We need to know the alphabet and we add a padding value "\0" to the alphabet.

# %%
alphabet = ["\0", *sorted(list(set(data)))]
n_alphabet = len(alphabet)
n_alphabet

# %%
char2index = {c: i for i, c in enumerate(alphabet)}
index2char = {i: c for i, c in enumerate(alphabet)}

# %% [markdown]
# Convert the data into a list of integers

# %%
index = [char2index[c] for c in data]

# %%
print(index[:25])
print("".join(index2char[i] for i in index[:25]))

# %%
index[0: 3]

# %%
X, y = [], []
for i in range(len(index) - 4):
    X.append(index[i : i + 3])
    y.append(index[i + 3])
    
X = np.stack(X)
y = np.stack(y)

# %%
X.shape, y.shape

# %%
X[0], y[0]

# %%
type(y)

# %%
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset


train_ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
train_dl = DataLoader(train_ds, batch_size=500)

# %% [markdown]
# # The model

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device


# %%
class CharModel(nn.Module):
    def __init__(self, n_vocab, n_embedding, n_hidden):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, n_embedding)
        self.lin_in = nn.Linear(n_embedding, n_hidden)
        
        self.lin_hidden = nn.Linear(n_hidden, n_hidden)
        self.lin_out = nn.Linear(n_hidden, n_vocab)
        
    def forward(self, X):
        c1, c2, c3 = X[:, 0], X[:, 1], X[:, 2]
        
        in1 = F.relu(self.lin_in(self.emb(c1)))
        h = F.tanh(self.lin_hidden(in1))
                   
        in2 = F.relu(self.lin_in(self.emb(c2)))
        h = F.tanh(self.lin_hidden(h + in2))
        
        in3 = F.relu(self.lin_in(self.emb(c3)))
        h = F.tanh(self.lin_hidden(h + in3))
        
        return F.log_softmax(self.lin_out(h), dim=-1)


# %%
n_embedding = 40
n_hidden = 256

model = CharModel(n_alphabet, n_embedding=40, n_hidden=128)
model = model.to(device)

# %%
optimizer = optim.Adam(model.parameters(), 0.001)
#criterion = nn.CrossEntropyLoss()
criterion = F.nll_loss


# %%
def fit(model, n_epoch=2):
    optimizer = optim.Adam(model.parameters(), 0.001)
    
    for epoch in range(n_epoch):
        print(f"Epoch {epoch}:")
        running_loss, correct = 0.0, 0

        model.train()
        for X, y in train_dl:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            y_ = model(X)
            loss = criterion(y_, y)

            loss.backward()
            optimizer.step()

            _, y_label_ = torch.max(y_, 1)
            correct += (y_label_ == y).sum().item()
            running_loss += loss.item() * X.shape[0]

        print(f"  Train Loss: {running_loss / len(train_dl.dataset):0.4f}")
        print(f"  Train Acc:  {correct / len(train_dl.dataset):0.2f}")


# %%
fit(model, 2)


# %%
def predict(word):
    word_idx = [char2index[c] for c in word]
    word_idx
    with torch.no_grad():
        X = torch.tensor(word_idx).unsqueeze(0).to(device)
        model.eval()
        y_ = model(X).cpu()
    pred = index2char[torch.argmax(y_).item()]
    print(f"{word} --> '{pred}'")


# %%
predict("the")

# %%
predict("wom")

# %%
predict("man")

# %%
predict("hum")


# %%
class CharModel(nn.Module):
    def __init__(self, n_vocab, n_embedding, n_hidden):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, n_embedding)
        self.lin_in = nn.Linear(n_embedding, n_hidden)
        self.lin_hidden = nn.Linear(n_hidden, n_hidden)
        self.lin_out = nn.Linear(n_hidden, n_vocab)
        
    def forward(self, X):
        c1, c2, c3 = X[:, 0], X[:, 1], X[:, 2]
        
        in1 = F.relu(self.lin_in(self.emb(c1)))       
        in2 = F.relu(self.lin_in(self.emb(c2)))
        in3 = F.relu(self.lin_in(self.emb(c3)))

        h = F.tanh(self.lin_hidden(in1))
        h = F.tanh(self.lin_hidden(h + in2))
        h = F.tanh(self.lin_hidden(h + in3))
        
        return F.log_softmax(self.lin_out(h), dim=-1)


# %%
model = CharModel(n_alphabet, n_embedding=n_embedding, n_hidden=128).to(device)
fit(model)

print()
predict("the")
predict("wom")
predict("man")
predict("hum")


# %%
class CharModel(nn.Module):
    def __init__(self, n_vocab, n_embedding, n_hidden):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, n_embedding)
        self.lin_in = nn.Linear(n_embedding, n_hidden)
        self.lin_hidden = nn.Linear(n_hidden, n_hidden)
        self.lin_out = nn.Linear(n_hidden, n_vocab)
        
        self.n_hidden = n_hidden
        
    def forward(self, X):
        c1, c2, c3 = X[:, 0], X[:, 1], X[:, 2]
        
        in1 = F.relu(self.lin_in(self.emb(c1)))       
        in2 = F.relu(self.lin_in(self.emb(c2)))
        in3 = F.relu(self.lin_in(self.emb(c3)))
        
        h = torch.zeros(X.shape[0], n_hidden, requires_grad=True).to(device)
        h = F.tanh(self.lin_hidden(h + in1))
        h = F.tanh(self.lin_hidden(h + in2))
        h = F.tanh(self.lin_hidden(h + in3))
        
        return F.log_softmax(self.lin_out(h), dim=-1)


# %%
model = CharModel(n_alphabet, n_embedding=n_embedding, n_hidden=n_hidden).to(device)
fit(model)

print()
predict("the")
predict("wom")
predict("man")
predict("hum")


# %%
class CharModel(nn.Module):
    def __init__(self, n_vocab, n_embedding, n_hidden):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, n_embedding)
        self.lin_in = nn.Linear(n_embedding, n_hidden)
        self.lin_hidden = nn.Linear(n_hidden, n_hidden)
        self.lin_out = nn.Linear(n_hidden, n_vocab)
        
        self.n_hidden = n_hidden
        
    def forward(self, X):
        h = torch.zeros(X.shape[0], n_hidden, requires_grad=True).to(device)
        for i in range(X.shape[1]):
            c = X[:, i]
            in_ = F.relu(self.lin_in(self.emb(c)))
            h = F.tanh(self.lin_hidden(h + in_))

        return F.log_softmax(self.lin_out(h), dim=-1)


# %%
model = CharModel(n_alphabet, n_embedding=n_embedding, n_hidden=n_hidden).to(device)
fit(model)

print()
predict("the")
predict("wom")
predict("man")
predict("hum")

# %%
predict("the huma")

# %%
predict("those ")

# %%
predict("those o")

# %%
predict("those of ")

# %%
predict("those of u")


# %% [markdown]
# You can use `nn.Sequential` to make it a bit more readable.

# %%
class CharModel(nn.Module):
    def __init__(self, n_vocab, n_embedding, n_hidden):
        super().__init__()
        self.i2e = nn.Sequential(
            nn.Embedding(n_vocab, n_embedding),
            nn.Linear(n_embedding, n_hidden),
            nn.ReLU(),
        )
        self.h2h = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
        )
        self.h2out = nn.Linear(n_hidden, n_vocab)
        
        self.n_hidden = n_hidden
        
    def forward(self, X):
        h = torch.zeros(X.shape[0], n_hidden, requires_grad=True).to(device)
        for i in range(X.shape[1]):
            c = X[:, i]
            h = self.h2h(h + self.i2e(c))

        return F.log_softmax(self.h2out(h), dim=-1)


# %%
model = CharModel(n_alphabet, n_embedding=n_embedding, n_hidden=n_hidden).to(device)
fit(model)

print()
predict("the")
predict("wom")
predict("man")
predict("hum")
