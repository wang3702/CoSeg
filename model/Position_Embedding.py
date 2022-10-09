import torch.nn as nn
import torch
import math
#classic absolute position embedding
# cos-sin positional embedding

class PositionalEmbedding(nn.Module):

    def __init__(self, embed_dimmension, max_len=512):

        """

        :param embed_dimmension:
        :param max_len:  the max length of the incoming sequence (default=5000).
        """
        super(PositionalEmbedding,self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, embed_dimmension).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        #shape study_length
        div_term = (torch.arange(0, embed_dimmension, 2).float() * -(math.log(10000.0) /embed_dimmension)).exp()
        #study_length*embedding_size
        pe[:, 0::2] = torch.sin(position * div_term)#1st chunk of 2 chunk
        pe[:, 1::2] = torch.cos(position * div_term)#2nd chunk of 2 chunk
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        #batch_size*seq_length*embedding_dim
        return self.pe[:, :x.size(1)]
