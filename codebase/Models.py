import torch
import torch.nn as nn
from torch.autograd import Variable

class PoseLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim, num_layers):
        super(PoseLSTM, self).__init__()

        ''' Constructor
        Arguments:
            input_dim: The number of expected features in the input
            hidden_dim: The desired number of features in the hidden state
            num_layers: Number of recurrent layers.
                E.g., setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM

        Note:
            Due to "batch_first = True",
            The first axis indexes instances in the mini_batch,
            the second is the sequence itself,
            and the third indexes elements of the input.
        '''

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Define the LSTM architecture
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first = True)
        # Define the output layer
        self.decoder = nn.Linear(self.hidden_dim, self.output_dim)

    def init_hidden(self):
        return (Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)).cuda(),
                Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)).cuda())

    def forward(self, inputs):
        ''' LSTM Forward step

        Arguments:
            input: input 3-D tensor with shape [batch_size, seq_length, features]
        '''

        lstm_output, self.hidden = self.lstm(inputs, self.hidden)
        y_pred = self.decoder(lstm_output.view((self.batch_size, -1, self.hidden_dim)))
        # y_pred = self.decoder(lstm_output[-1])
        return y_pred.view((self.batch_size, -1, self.output_dim))
