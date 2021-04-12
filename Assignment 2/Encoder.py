import torch
import torch.nn as nn

from ConvS2S import ConvEncoder
from Attention import MultiHeadAttention, PositionFeedforward

class Encoder(nn.Module): # 1 Mark
    def __init__(self, conv_layers, hidden_dim, feed_forward_dim=2048):
        super(Encoder, self).__init__()
        # Your code here
        
        self.conv=ConvEncoder(input_dim=hidden_dim,num_layers=conv_layers)
        self.feed_forward=PositionFeedforward(hidden_dim,feed_forward_dim)
        self.attention=MultiHeadAttention(hid_dim=hidden_dim,n_heads=16)



    def forward(self, input):
        """
        Forward Pass of the Encoder Class
        :param input: Input Tensor for the forward pass. 
        """
        # Your code here
        output = self.conv.forward(input)
        output = self.attention.forward(output,output,output)
        output = self.feed_forward.forward(output)
        return output

