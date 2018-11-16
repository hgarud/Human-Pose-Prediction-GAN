import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from Data import PennActionData
from Models import PoseLSTM
from utils import mse_measure, r2_measure

"""
Set baselines for the prediction task
* LSTM
* HMM-GMM
* R-GAN
"""

device = torch.device('cuda')

# Model Hyper-Parameters
hidden_dim = 128
num_layers = 2
seq_length = 16
n_epochs = 200
alpha = 1e-2

# Load saved Numpy file
sequences = np.load('All_16SL_26F_Sequences.npy')
data_len = len(sequences)
print(sequences.shape)
np.random.shuffle(sequences)

# Train-Validation split
train_size = 0.80
split_index = math.ceil(data_len * train_size) # 122869 samples
train_sequences = sequences[0:split_index]
validation_sequences = sequences[split_index:]


input_dim = sequences.shape[-1]         # 26 when withVisibility = False
                                        # 39 when withVisibility = True

output_dim = input_dim
batch_size = 6553
total_train_steps = len(train_sequences)/batch_size
total_valid_steps = len(validation_sequences)/batch_size

# LSTM
model = PoseLSTM(input_dim = input_dim, hidden_dim = hidden_dim,
                batch_size = batch_size, output_dim = output_dim, num_layers = num_layers)

# Initialise hidden state
model.hidden = model.init_hidden()
model.cuda()

loss_fn = nn.MSELoss(reduction = 'elementwise_mean')
optimizer = optim.SGD(model.parameters(), lr = alpha, momentum = 0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer, mode = 'min', factor = 0.5, verbose = True)

loss = 0
train_losses = []
valid_losses = []
explained_variances = []
r2s = []
mses = []
try:
    print("Training for {} epochs...".format(n_epochs))
    for epoch in range(1, n_epochs + 1):
        mean_epoch_train_loss = 0
        for batch_index in range(0, train_sequences.shape[0], batch_size):
            X_train = torch.from_numpy(train_sequences[batch_index:batch_index+batch_size,:-1,:]).float().to(device)
            X_train = X_train.view((-1, seq_length - 1, input_dim))

            y_train = torch.from_numpy(train_sequences[batch_index:batch_index+batch_size,1:,:]).float().to(device)
            y_train = y_train.view((-1, seq_length - 1, input_dim))

            # Clear stored gradient
            model.zero_grad()

            # Forward pass
            y_pred = model.forward(X_train)

            train_loss = loss_fn(y_pred, y_train)
            mean_epoch_train_loss = mean_epoch_train_loss+ train_loss.item()
            print("Loss over {} sequences: {} for step {}/{} @ epoch #{}".format(data_len, train_loss, (batch_index/batch_size) + 1, total_train_steps, epoch))

            # Backward and optimize
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        train_losses.append(mean_epoch_train_loss/total_train_steps)

        mean_epoch_valid_loss = 0
        mean_epoch_valid_r2 = 0
        mean_epoch_valid_mse = 0
        for batch_index in range(0, validation_sequences.shape[0], batch_size):
            X_valid = torch.from_numpy(validation_sequences[batch_index:batch_index+batch_size,:-1,:]).float().to(device)
            X_valid = X_valid.view((-1, seq_length - 1, input_dim))

            y_valid = torch.from_numpy(validation_sequences[batch_index:batch_index+batch_size,1:,:]).float().to(device)
            y_valid = y_valid.view((-1, seq_length - 1, input_dim))

            # Forward pass
            y_pred = model.forward(X_valid)

            valid_loss = loss_fn(y_pred, y_valid)
            mean_epoch_valid_mse = mean_epoch_valid_mse + mse_measure(y_pred, y_valid)
            mean_epoch_valid_r2 = mean_epoch_valid_r2 + r2_measure(y_pred, y_valid)
            mean_epoch_valid_loss = mean_epoch_valid_loss+ valid_loss.item()
        valid_losses.append(mean_epoch_valid_loss/total_valid_steps)
        r2s.append(mean_epoch_valid_r2/total_valid_steps)
        mses.append(mean_epoch_valid_mse/total_valid_steps)

        scheduler.step(valid_losses[-1])
except Exception as e:
    print("Exiting with exception: ", e)
    # Save the model checkpoint
    checkpoint_file = "LSTMmodel_epoch" + str(epoch) + "_valid_loss" + str(valid_losses[-1]) + ".ckpt"
    torch.save(model.state_dict(), checkpoint_file)

# Save the model checkpoint
checkpoint_file = "Visibility_LSTMmodel_epoch" + str(epoch) + "_valid_loss" + str(valid_losses[-1]) + ".ckpt"
torch.save(model.state_dict(), checkpoint_file)

fig = plt.figure()
plt.plot(valid_losses, 'k')
plt.plot(train_losses, 'r')

fig = plt.figure()
plt.plot(r2s, 'b')
plt.plot(mses, 'g')
plt.show()
