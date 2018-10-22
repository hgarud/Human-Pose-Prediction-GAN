import torch
import torch.nn as nn
from torch.autograd import Variable

class PoseLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim, num_layers):
        """ Constructor
        Args:
            input_dim: The number of expected features in the input
            hidden_dim: The desired number of features in the hidden state
            num_layers: Number of recurrent layers.
                E.g., setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM

        Note:
            Due to "batch_first = True",
            The first axis indexes instances in the mini_batch,
            the second is the sequence itself,
            and the third indexes elements of the input.
        """
        super(PoseLSTM, self).__init__()

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
        """ LSTM Forward step

        Arguments:
            input: input 3-D tensor with shape [batch_size, seq_length, features]
        """

        lstm_output, self.hidden = self.lstm(inputs, self.hidden)
        y_pred = self.decoder(lstm_output.view((self.batch_size, -1, self.hidden_dim)))
        # y_pred = self.decoder(lstm_output[-1])
        return y_pred.view((self.batch_size, -1, self.output_dim))

class GeneratorLSTM(PoseLSTM):
    """
    The Generator base class for PoseRGAN.
    Inherits PoseLSTM because of same structure.

    Args:
        input_dim (int): The number of expected features in the input
        hidden_dim (int): The desired number of features in the hidden state
        num_layers (int): Number of recurrent layers.
            E.g., setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM

    """

    def __getattribute__(self, name):
        if name in ['init_hidden']: raise AttributeError(name)
        else: return super(Generator, self).__getattribute__(name)

    def __dir__(self):
        return sorted((set(dir(self.__class__)) | set(self.__dict__.keys())) — set(['init_hidden']))

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim, num_layers):
        super(Generator, self).__init__(input_dim, hidden_dim, batch_size, output_dim, num_layers)

class DiscriminatorLSTM(PoseLSTM):
    """
    The Discriminator base class for PoseRGAN.
    Inherits PoseLSTM because of same structure.

    Args:
        input_dim (int): The number of expected features in the input
        hidden_dim (int): The desired number of features in the hidden state
        num_layers (int): Number of recurrent layers.
            E.g., setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM

    """

    def __getattribute__(self, name):
        if name in ['init_hidden']: raise AttributeError(name)
        else: return super(Discriminator, self).__getattribute__(name)

    def __dir__(self):
        return sorted((set(dir(self.__class__)) | set(self.__dict__.keys())) — set(['init_hidden']))

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim, num_layers):
        super(Discriminator, self).__init__(input_dim, hidden_dim, batch_size, output_dim, num_layers)


class DiscriminatorCNN(nn.Module):
    def __init__(self):
        """
        A second discrominator for time series prediction as detailed in Xingyu Zhou et al.
        A CNN for deciding whether the input vector is a genuine data sample.

        Reference:  https://doi.org/10.1155/2018/4907423

        Args:
            variable (type): description

        Returns:
            type: description

        Raises:
            Exception: description

        """
        super(DiscriminatorCNN, self).__init__()
        raise NotImplementedError

class PoseRGAN(object):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim, num_layers):
        """ Constructor
        Arguments:
            input_dim (int): The number of expected features in the input
            hidden_dim (int): The desired number of features in the hidden state
            num_layers (int): Number of recurrent layers.
                E.g., setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM

        Note:
            Due to "batch_first = True",
            The first axis indexes instances in the mini_batch,
            the second is the sequence itself,
            and the third indexes elements of the input.
        """
        # super(PoseRGAN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.num_layers = num_layers

    def init_normal_hidden(self, mu, sigma):
        # torch.nn.init.normal_(self.hidden.data, mu, sigma)
        raise NotImplementedError

    def latent_space_sampler(self):
        raise NotImplementedError
