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
# # Debugging

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# **Q: "No debugger for your code. What do you think?"**
#
# **A: "I would NOT be able to code!"**
#
# - Who does "print-line-debugging"?
# - Who likes debugging in tensorflow?
# - What is the intersection of those two groups?
#
#
# ## IPDB cheatsheet
# IPython Debugger
#
# Taken from http://wangchuan.github.io/coding/2017/07/12/ipdb-cheat-sheet.html
#
# - h(help): Print help
#
# - n(ext): Continue execution until the next line in the current function is reached or it returns.
# - s(tep): Execute the current line, stop at the first possible occasion (either in a function that is called or in the current function).
# - r(eturn): Continue execution until the current function returns.
# - c(ont(inue)): Continue execution, only stop when a breakpoint is encountered.
#
# - r(eturn): Continue execution until the current function returns.
# - a(rgs): Print the argument list of the current function.

# %% [markdown]
# Note: Python 3.7 has `breakpoint()` built-in! [[PEP 553]](https://www.python.org/dev/peps/pep-0553/)

# %%
from IPython.core.debugger import set_trace


# %%
def my_function(x):
    answer = 42
    # set_trace()  # <-- uncomment!
    answer += x
    return answer

my_function(12)

# %% [markdown]
# ## Example: debuging a NN

# %%
X = torch.rand((5, 3))
X


# %%
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3, 1)
    
    def forward(self, X):
        # set_trace()
        x = self.lin(X)
        return X

    
model = MyModule()
y_ = model(X)

# assert y_.shape == (5, 1), y_.shape

# %% [markdown]
# ## Debug Layer

# %%
class DebugModule(nn.Module):
    def forward(self, x):
        set_trace()
        return x


# %%
model = nn.Sequential(
    nn.Linear(1, 5),
    DebugModule(),
    nn.Linear(5, 1),
)

# %%
X = torch.unsqueeze(torch.tensor([1.]), dim=0)
# model(X)

# %% [markdown]
# ## Tensorboard and `tensorboardX`
# Tensorboard and `tensorboardX` are also great to debug a model, e.g. to look at the gradients.
