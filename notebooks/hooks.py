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
# # Hooks
# Hooks are simple functions that can be registered to be called during the forward or backward pass of a `nn.Module`.
# These functions can be used to print out information or modify the module.
#
# Here is a simple forward hook example that prints some information about the input and output of a module.
#
# Tip: Don't forget to remove the hook afterwards!
#
# Ref:
# - https://pytorch.org/tutorials/beginner/former_torchies/nn_tutorial.html#forward-and-backward-function-hooks
# - https://pytorch.org/docs/master/nn.html#torch.nn.Module.register_forward_hook
# - https://pytorch.org/docs/master/nn.html#torch.nn.Module.register_forward_pre_hook
# - https://pytorch.org/docs/master/nn.html#torch.nn.Module.register_backward_hook

# %%
def tensorinfo_hook(module, input_, output):
    """
    Register this forward hook to print some infos about the tensor/module.

    Example:

        >>> from torchvision.models import resnet18
        >>> model = resnet18(pretrained=False)
        >>> hook_fc = model.fc.register_forward_hook(tensorinfo_hook)
        >>> # model(torch.ones(1, 3, 244, 244))
        >>> hook_fc.remove()

    """
    print(f"Inside '{module.__class__.__name__}' forward")
    print(f"  input:     {str(type(input_)):<25}")
    print(f"  input[0]:  {str(type(input_[0])):<25} {input_[0].size()}")
    print(f"  output:    {str(type(output)):<25} {output.data.size()}")
    print()


# %%
import torch
import torch.nn as nn

# %%
m = nn.Linear(1, 3)

# %%
hook = m.register_forward_hook(tensorinfo_hook)

# %%
m(torch.rand(1));

# %%
hook.remove()

# %% [markdown]
# ## Exercise
# - Write a context manager hook that removes the hook when leaving the with block.
