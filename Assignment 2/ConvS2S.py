import torch.nn as nn

class ConvEncoder(nn.Module):
    def __init__(self, input_dim, num_layers=3):
        """
        The convolutional Encoder layer.
        Model based on the paper "Convolutional Sequence to Sequence Learning - https://arxiv.org/pdf/1705.03122.pdf"
        :param input_dim: input dimension of the tensor
        :param num_layers: Number of convolutional you desire to stack on top of each other
        """
        super(ConvEncoder, self).__init__()
        self.convolutions = self.get_convolutions(input_dim=input_dim, num_layers=num_layers)

    def forward(self, source): # 1 Mark
        """
        Remember, we have multiple convolutional layers, for iterate over them.
        :param source: Input tensor for the forward pass. 
        """
        for conv in self.convolutions:
            # Your code goes here.
            res = conv(source)
            res = nn.functional.glu(res, dim=1)
            # Your code goes here.
            obtained_output=res+source
            source=obtained_output
        return source

    def get_convolutions(self, input_dim, num_layers=3): # 0.5 Marks
        """
        :param input_dim: input dimension
        :param num_layers: Number of convolutional you desire to stack on top of each other
        :return: nn.ModuleList()
        """
        list_of_conv1d_layers=nn.ModuleList([nn.Conv1d(input_dim,2*input_dim,kernel_size=3,stride=1,padding=1) for i in range(num_layers)])
        return list_of_conv1d_layers
        
        
