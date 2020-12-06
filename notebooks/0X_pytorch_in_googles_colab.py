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
# # Using PyTorch + GPU/TPU in Google's Colab
#
# > Colaboratory is a Google research project created to help disseminate machine learning education and research. It's a Jupyter notebook environment that requires no setup to use and runs entirely in the cloud.
# > Colaboratory notebooks are stored in Google Drive and can be shared just as you would with Google Docs or Sheets. Colaboratory is free to use.
# > -- https://colab.research.google.com/notebooks/welcome.ipynb
#
# **Setup**
# - Go to https://colab.research.google.com
# - Create a new python 3 notebook
# - Enable the GPU: "Edit -> Notebook settings -> Hardware accelerator: GPU -> Save"
# - Then try the following:
#
# ```python
# import torch
#
# print(torch.__version__)
#
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# print(DEVICE)
# ```

# %% [markdown]
# You should get something like this:
# > 1.0.1.post2
# >
# > cuda

# %% [markdown]
# # Using this Repo in Colab
# You can use this repo with google colab,
# but not all notebooks run without changes.
# Some notebooks import from `utils.py` which is not availbale on colab.
# You have to remove that line and copy and paste the required function/class into the notebook.
#
# It's easy to use colab. Simply append the url from the notebook on github to `https://colab.research.google.com/github/`. E.g. `notebooks/pytorch_basics.ipynb` is available under:
# https://colab.research.google.com/github/sotte/pytorch_tutorial/blob/master/notebooks/pytorch_basics.ipynb)
