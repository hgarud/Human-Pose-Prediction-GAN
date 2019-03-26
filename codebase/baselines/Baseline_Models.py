import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import sys
sys.path.append("../")
from IronHide import NoamOpt
from Models import PoseLSTM
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda')

class ERD(nn.Module):
    """
    Encoder-Recurrent-Decoder architecture for Human Pose Forecasting.

    Reference: https://arxiv.org/pdf/1508.00271.pdf

    Args:
        variable (type): description

    Returns:
        type: description

    Raises:
        Exception: description

    """

    def __init__(self, seq_length=16, input_dim=39, enc_hidden_dim=[512, 512], dec_hidden_dim = [500, 100], rec_hidden_dim=1000, batch_size=32, output_dim=39, num_layers=2, dropout=0.1):
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
        super(ERD, self).__init__()

        self.input_dim = input_dim
        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.rec_hidden_dim = rec_hidden_dim
        self.seq_length = seq_length
        self.dropout = nn.Dropout(p=dropout)

        # Encoder
        self.enc_fc1 = nn.Linear(self.input_dim, self.enc_hidden_dim[0])
        self.enc_fc2 = nn.Linear(self.enc_hidden_dim[0], self.enc_hidden_dim[1])

        # Recurrent
        self.lstm = nn.LSTM(self.enc_hidden_dim[1], self.rec_hidden_dim, self.num_layers, batch_first = True)
        # self.lstm_1 = nn.LSTMCell(self.enc_hidden_dim[1], self.rec_hidden_dim)
        # self.lstm_2 = nn.LSTMCell(self.rec_hidden_dim, self.rec_hidden_dim)

        # Decoder
        self.dec_fc1 = nn.Linear(self.rec_hidden_dim, self.dec_hidden_dim[0])
        self.dec_fc2 = nn.Linear(self.dec_hidden_dim[0], self.dec_hidden_dim[1])
        self.dec_fc3 = nn.Linear(self.dec_hidden_dim[1], self.output_dim)

    def init_hidden(self):
        return (Variable(torch.zeros(self.num_layers, self.batch_size, self.rec_hidden_dim, dtype = torch.float)).cuda(),
                Variable(torch.zeros(self.num_layers, self.batch_size, self.rec_hidden_dim, dtype = torch.float)).cuda())

    def repackage_hidden(self):
        """
        Wraps hidden states in new Variables, to detach them from their history.
        Results in much faster training times and no memory leakage. :)
        """
        return tuple(Variable(var) for var in self.hidden)

    def encode(self, inputs):
        x = self.dropout(F.relu(self.enc_fc1(inputs)))
        x = self.dropout(self.enc_fc2(x))

        return x

    def decode(self, inputs):
        x = self.dropout(F.relu(self.dec_fc1(inputs)))
        x = self.dropout(F.relu(self.dec_fc2(x)))
        x = F.relu(self.dec_fc3(x))

        return x

    def forward_1(self, inputs):
        # empty tensor for the output of the lstm
        output_seq = torch.empty((self.batch_size,
                                    inputs.shape[1],
                                    self.output_dim)).cuda()

        # pass the hidden and the cell state from one lstm cell to the next one
        # we also feed the output of the first layer lstm cell at time step t to the second layer cell
        # init the both layer cells with the zero hidden and zero cell states
        hidden = self.hidden

        # for every step in the sequence
        for t in range(inputs.shape[1]):

            x = self.encode(inputs[:, t, :])
            # get the hidden and cell states from the first layer cell
            hidden_1 = self.lstm_1(x, hidden[0])

            # unpack the hidden and the cell states from the first layer
            # h_1, c_1 = hc_1

            # pass the hidden state from the first layer to the cell in the second layer
            hidden_2 = self.lstm_2(hidden_1[0], hidden[1])

            # unpack the hidden and cell states from the second layer cell
            # h_2, c_2 = hc_2

            output_seq[:, t, :] = self.decode(hidden_2[0])
            # form the output of the fc


        # return the output sequence
        return output_seq

    def forward(self, inputs):
        """ LSTM Forward step

        Arguments:
            input: input 3-D tensor with shape [batch_size, seq_length, features]
        """
        x = self.encode(inputs)
        x, self.hidden = self.lstm(x, self.hidden)
        x = self.decode(x)
        self.hidden = self.repackage_hidden()
        return x

class LSTM3LR(PoseLSTM):
    """
    The child class for LSTM-3LR network.
    Inherits PoseLSTM because of same structure.

    Args:
        input_dim (int): The number of expected features in the input
        hidden_dim (int): The desired number of features in the hidden state
        num_layers (int): Number of recurrent layers.
            E.g., setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM
    """
    def __init__(self, input_dim = 39, hidden_dim = 26, batch_size = 32, output_dim = 39, num_layers = 1, dropout = 0.0):
        super(LSTM3LR, self).__init__(input_dim, hidden_dim, batch_size, output_dim, num_layers, dropout)

if __name__ == '__main__':

    # X = torch.rand(32, 16, 26).cuda()
    sequences = np.load('../Normalized_Train_16SL_26F_Sequences.npy')
    data_len = len(sequences)                       # 68672
    # print(sequences.shape)
    np.random.shuffle(sequences)

    input_dim = sequences.shape[-1]
    output_dim = input_dim
    batch_size = 64
    total_train_steps = data_len//batch_size
    seq_length = 16
    dropout = 0.1

    # erd = ERD(seq_length=16, input_dim=input_dim, output_dim=output_dim, batch_size=batch_size, dropout=0.2)
    # erd.hidden = erd.init_hidden()
    # erd.train().cuda()
    model = LSTM3LR(input_dim=input_dim, hidden_dim=1000, output_dim=output_dim, num_layers=3, batch_size=batch_size, dropout=dropout)
    model.hidden = model.init_hidden()
    model.train().cuda()
    # erd.eval().cuda()

    model_opt = NoamOpt(model.hidden_dim, 1, 2900,
                    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    # model_opt = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)

    mse_loss_fn = nn.MSELoss()

    n_epochs = 6

    train_losses = []
    # valid_losses = []
    print("Training for {} epochs...".format(n_epochs))
    for epoch in range(1, n_epochs + 1):
        for batch_index in range(0, sequences.shape[0], batch_size):
            X_train = torch.from_numpy(sequences[batch_index:batch_index+batch_size,:8,:]).float().to(device)
            X_train = X_train.view((-1, 8, input_dim))

            y_train = torch.from_numpy(sequences[batch_index:batch_index+batch_size,8:,:]).float().to(device)
            y_train = y_train.view((-1, 8, input_dim))

            model.zero_grad()
            out = model.forward(X_train)
            # print("#######################################", out.shape)
            # out = erd.generator(out)
            loss = mse_loss_fn(out, y_train)
            train_losses.append(loss.item())
            print("Loss over {} sequences: {} for step {}/{} @ epoch #{}/{}".format(data_len,
                                    loss, (batch_index/batch_size) + 1,
                                    total_train_steps, epoch, n_epochs))
            model_opt.optimizer.zero_grad()
            # model_opt.zero_grad()
            loss.backward()
            model_opt.step()

    # y = erd(X)
    # print(y.shape)
    fig = plt.figure()
    # plt.plot(valid_losses, 'k', label = 'validation loss')
    plt.plot(train_losses, 'r', label = 'training loss')
    plt.show()

    checkpoint_file = "LSTM_3LR_DO" + str(dropout) + "_loss_" + str(train_losses[-1]) + ".ckpt"
    torch.save(model.state_dict(), checkpoint_file)
