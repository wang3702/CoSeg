
import torch.nn as nn
from model.Position_Embedding import PositionalEmbedding
from model.TransformerBlock import TransformerBlock
import torch
class MaskedVideoModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden_size,embedding_length,study_length,n_layers,attn_heads=12,dropout=0.1):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        :param study_length: study video chunk length
        :param attn_heads:number of attention heads
        """

        super(MaskedVideoModel,self).__init__()
        self.position_embedding=PositionalEmbedding(embedding_length,study_length)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, attn_heads, hidden_size * 4, dropout) for _ in range(n_layers)])
        self.softmax = nn.LogSoftmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(hidden_size, embedding_length)
    def forward(self, x,mask_label):
        #position embedding

        x = x+self.position_embedding(x)
        # attention masking for padded token
        # mask:torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        # mask label: batch_size* seq length 0 denotes mask
        mask = (mask_label > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        x=self.dropout(x)
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        x=self.linear(x)

        return x
