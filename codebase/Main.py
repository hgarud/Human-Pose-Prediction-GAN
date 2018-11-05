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
batch_size = 47
num_layers = 2
seq_length = 16
n_epochs = 300
alpha = 1e-2

data_stream = PennActionData(base_dir = '/home/hrishi/1Hrishi/0Thesis/Data/Penn_Action/labels/', file = '0129.mat', scaling = 'standard')
data_len = data_stream.data_len
print(data_len)
sequences = data_stream.getSequences(seq_length = seq_length, withVisibility = False)
# sequences.cuda()

input_dim = sequences.shape[-1]         # 26 when withVisibility = False
                                        # 39 when withVisibility = True

output_dim = input_dim

# LSTM
model = PoseLSTM(input_dim = input_dim, hidden_dim = hidden_dim,
                batch_size = batch_size, output_dim = output_dim, num_layers = num_layers)
# Initialise hidden state
model.hidden = model.init_hidden()
# model.double().cuda()
model.cuda()
# print(model.device())
loss_fn = nn.MSELoss(reduction = 'elementwise_mean')
optimizer = optim.SGD(model.parameters(), lr = alpha)

try:
# average_losses = []
    losses = []
    print("Training for {} epochs...".format(n_epochs))
    for epoch in tqdm(range(1, n_epochs + 1)):
        # Clear stored gradient
        model.zero_grad()
        loss = 0
        # for i in range(0, sequences.shape[1] - 1):
            # Get Random Training Sequence
            # X_train, y_train = data_stream.get_random_training_set(seq_len = seq_length, batch_size = batch_size)
            # X_train.cuda()
            # y_train.cuda()

        X_train = torch.from_numpy(sequences[:,:-1,:]).float().to(device)
        # X_train.to(device)
        X_train = X_train.view((batch_size, -1, input_dim))
            #
        y_train = torch.from_numpy(sequences[:,1:,:]).float().to(device)
        # y_train.to(device)
        y_train = y_train.view((batch_size, -1, input_dim))
            # # Forward pass
            # y_pred = model.forward(X_train)
        # sequences = torch.from_numpy(sequences).float()
        # sequences.cuda()
        # X_train = X_train.view((batch_size, -1, input_dim))

        # y_train = torch.from_numpy(sequences[:,1:,:]).float()
        # y_train.cuda()
        # y_train = y_train.view((batch_size, -1, input_dim))
        # Forward pass
        y_pred = model.forward(X_train)

            ## Sanity Checks :p
            # print("Complete Sequence shape: ", sequence.shape)
            # print("Training Sequence shape: ", sequence[:,:-1].shape)
            # print("Target Sequence shape: ", sequence[:,1:].shape)
            # print("Predicted Sequence shape: ", y_pred.shape)

            # Calculate Loss
            # loss = loss + loss_fn(y_pred, y_train)
        loss = loss_fn(y_pred, y_train)
        # av_epoch_loss = loss.item()//seq_length
        # print("Average loss over {} sequences: {} @ epoch #{}".format(data_len, av_epoch_loss, epoch))
        print("Loss over {} sequences: {} @ epoch #{}".format(data_len, loss, epoch))
        # average_losses.append(av_epoch_loss)
        losses.append(loss)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward(retain_graph = True)
        optimizer.step()
except:
    print("Exiting")

    # Save the model checkpoint
    checkpoint_file = "model_epoch" + str(epoch) + "_train_loss" + str(losses[-1]) + "_lr" + str(alpha) + ".ckpt"
    torch.save(model.state_dict(), checkpoint_file)

import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(losses, 'k')

# checkpoint = torch.load('model.ckpt')
# model.load_state_dict(checkpoint)

# Test the model
explained_variances = []
r2s = []
mses = []
with torch.no_grad():
    from sklearn.metrics import explained_variance_score, r2_score, mean_squared_error
    # Get random training sample
    # rand_key = np.random.randint(low = 0, high = data_len)
    # X_test = sequences[]
    X_test = torch.from_numpy(sequences[:,:-1,:]).float().to(device)
    # sequence = sequence.cuda()
    X_test = X_test.view((batch_size, -1, input_dim))

    y_test = torch.from_numpy(sequences[:,1:,:]).float().to(device)
    y_test = y_test.view((batch_size, -1, input_dim))
    # Evaluate
    y_pred = model.forward(X_test)
    print(y_pred.shape)
    # rand_key = np.random.randint(low = 0, high = data_len)

    # for i in range(len(y_pred)):
    #     explained_variances.append(explained_variance_score(y_test[i, :, :], y_pred[i, :, :]))
    #     r2s.append(r2_score(y_test[i, :, :], y_pred[i, :, :]))
    #     mses.append(mean_squared_error(y_test[i, :, :], y_pred[i, :, :]))

    for i in range(seq_length-1):
        explained_variances.append(explained_variance_score(y_test[:, i, :], y_pred[:, i, :]))
        r2s.append(r2_score(y_test[:, i, :], y_pred[:, i, :]))
        mses.append(mean_squared_error(y_test[:, i, :], y_pred[:, i, :]))

# # Save the model checkpoint
# torch.save(model.state_dict(), 'model.ckpt')

fig = plt.figure()
# frames = linspace(1, seq_length)
plt.plot(explained_variances, 'r')
plt.plot(r2s, 'b')
plt.plot(mses, 'g')
plt.show()
