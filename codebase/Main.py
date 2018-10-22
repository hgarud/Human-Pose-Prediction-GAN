from Data import PennActionData
import numpy as np
import torch.nn as nn
import torch.optim as optim
from Models import PoseLSTM
from tqdm import tqdm
import torch

"""
Set baselines for the prediction task
* LSTM
* HMM-GMM
* R-GAN
"""

# device = torch.device('cuda')

# Model Hyper-Parameters
hidden_dim = 26
batch_size = 1
output_dim = 39
num_layers = 1
seq_length = 16
input_dim = 39
n_epochs = 5

data_stream = PennActionData(base_dir = '/home/hrishi/1Hrishi/0Thesis/Data/Penn_Action/labels/', file = '0758.mat')
data_len = data_stream.data_len
sequences = data_stream.get_sequence_dict(seq_length = seq_length)
# sequences
# LSTM
model = PoseLSTM(input_dim = input_dim, hidden_dim = hidden_dim,
                batch_size = batch_size, output_dim = output_dim, num_layers = num_layers)
# Initialise hidden state
model.hidden = model.init_hidden()
model.cuda()
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 1e-5)

# try:
print("Training for {} epochs...".format(n_epochs))
for epoch in tqdm(range(1, n_epochs + 1)):
    # Clear stored gradient
    model.zero_grad()

    for _, sequence in sequences.items():
        # Get Random Training Sequence
        # X_train, y_train = data_stream.get_random_training_set(seq_len = seq_length, batch_size = batch_size)
        # X_train.cuda()
        # y_train.cuda()

        sequence = torch.from_numpy(sequence).float()
        sequence = sequence.cuda()
        sequence = sequence.view((batch_size, -1, input_dim))

        # Forward pass
        y_pred = model.forward(sequence[:,:-1])

        # Calculate Loss
        loss = loss_fn(y_pred, sequence[:,1:])

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
# except:
#     print("Exiting")
