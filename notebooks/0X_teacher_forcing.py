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
# # "Teacher Forcing"
#
# "Teacher forcing" is a method used in sequence2sequence models.
# It replaces a wrong words in the predicted sequence with the correct one.
#
# Think of a teacher that corrects your translation as soon as you say a wrong word to prevent you going off on a tangent.

# %% [markdown]
# Here is the pseudo code for teacher forcing:
#
# ```python
# class Seq2SeqModel(nn.Module):
#     def __init__(self, p_teacher_forcing: float):
#         self.p_teacher_forcing = p_teacher_forcing
#         # ...
#     
#     def forward(self, X, y):
#         # ... some calculation
#         current_word = torch.zeros(...)
#         result = []
#         for i in range(self.sentence_length):
#             # ... some calculation with current_word
#             result.append(output)
#             current_word = torch.argmax(output)
#             
#             # teacher forcing
#             if self.p_teacher_forcing > random.random():
#                 current_word = y[i]
#         
#         return torch.stack(result)
# ```
#
# Reduce `p_teacher_forcing` during training and let it converge to 0.
