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

device = torch.device('cuda')

# Model Hyper-Parameters
hidden_dim = 128
num_layers = 2
seq_length = 16
n_epochs = 500
alpha = 1e-2

data_stream = PennActionData(base_dir = '/home/hrishi/1Hrishi/0Thesis/Data/Penn_Action/labels/', file = '0129.mat', scaling = 'standard')
data_len = data_stream.data_len
print(data_len)
sequences = data_stream.getStridedSequences(seq_length = seq_length, withVisibility = False)
np.random.shuffle(sequences)

input_dim = sequences.shape[-1]         # 26 when withVisibility = False
                                        # 39 when withVisibility = True

output_dim = input_dim
batch_size = sequences.shape[0]

# LSTM
model = PoseLSTM(input_dim = input_dim, hidden_dim = hidden_dim,
                batch_size = batch_size, output_dim = output_dim, num_layers = num_layers)

# Initialise hidden state
model.hidden = model.init_hidden()
model.cuda()

loss_fn = nn.MSELoss(reduction = 'elementwise_mean')
optimizer = optim.SGD(model.parameters(), lr = alpha)

X_train = torch.from_numpy(sequences[:,:-1,:]).float().to(device)
X_train = X_train.view((batch_size, -1, input_dim))

y_train = torch.from_numpy(sequences[:,1:,:]).float().to(device)
y_train = y_train.view((batch_size, -1, input_dim))

loss = 0
try:
    losses = []
    print("Training for {} epochs...".format(n_epochs))
    for epoch in tqdm(range(1, n_epochs + 1)):
        # Clear stored gradient
        model.zero_grad()
        loss = 0

        # Forward pass
        y_pred = model.forward(X_train)

        loss = loss_fn(y_pred, y_train)
        losses.append(loss)
        print("Loss over {} sequences: {} @ epoch #{}".format(data_len, loss, epoch))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward(retain_graph = True)
        optimizer.step()
except:
    print("Exiting")
    # Save the model checkpoint
    checkpoint_file = "model_epoch" + str(epoch) + "_train_loss" + str(losses[-1]) + "_lr" + str(alpha) + ".ckpt"
    torch.save(model.state_dict(), checkpoint_file)

# Save the model checkpoint
checkpoint_file = "model_epoch" + str(epoch) + "_train_loss" + str(loss) + "_lr" + str(alpha) + ".ckpt"
torch.save(model.state_dict(), checkpoint_file)

import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(losses, 'k')

# Test the model
explained_variances = []
r2s = []
mses = []
with torch.no_grad():
    from sklearn.metrics import explained_variance_score, r2_score, mean_squared_error
    X_test = torch.from_numpy(sequences[:,:-1,:]).float().to(device)
    X_test = X_test.view((batch_size, -1, input_dim))

    y_test = torch.from_numpy(sequences[:,1:,:]).float().to(device)
    y_test = y_test.view((batch_size, -1, input_dim))

    # Evaluate
    y_pred = model.forward(X_test)
    print(y_pred.shape)

    for i in range(seq_length-1):
        explained_variances.append(explained_variance_score(y_test[:, i, :], y_pred[:, i, :]))
        r2s.append(r2_score(y_test[:, i, :], y_pred[:, i, :]))
        mses.append(mean_squared_error(y_test[:, i, :], y_pred[:, i, :]))

fig = plt.figure()
plt.plot(explained_variances, 'r')
plt.plot(r2s, 'b')
plt.plot(mses, 'g')
plt.show()
