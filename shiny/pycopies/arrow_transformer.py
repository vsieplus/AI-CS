# implementation of an 'arrow' transformer model with pytorch
# architecture ideas drawn from
#   transformers/other attention based mechanisms in nlp: arxiv.org/abs/1706.03762
#   music transformer: arxiv.org/abs/1809.04281

# Model Description (see paper for more detailed/formal definition)

import torch
import torch.nn as nn

class ArrowTransformer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self):
        pass
