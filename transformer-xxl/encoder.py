import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class Encoder(nn.Module()):
    def __init__(self, seqdim, seqlen, batch_size):
        """

        :param seqdim: the number of rows in your input data (data.shape[0])
        :param seqlen: the number of tokens in each row of your input data (data.shape[1])
        :param batch_size: the number of input examples to pass at a time to the model
        """

        self.seqdim = seqdim
        self.seqlen = seqlen
        self.batch_size = batch_size

        self.conv_enc = nn.Conv2d(seqdim, 1, 1)
        self.layer_norm_enc = nn.LayerNorm((1, seqlen))
        self.relu = nn.ReLU(inplace=True)
        self.bert = BertModel.from_pretrained("bert-base-cased")

    def forward(self, x, mask=None):

        x = self.conv_enc(x)
        x = self.layer_norm_enc(x)
        x = self.relu(x)
        x = x.view(self.batch_size, self.seqlen).long()
        x = self.bert(input_ids=x, attention_mask=mask)
        x = x[0]  # returning just the hidden states
        return x
