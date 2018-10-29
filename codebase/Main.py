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
hidden_dim = 128
batch_size = 1
output_dim = 39
num_layers = 2
seq_length = 16
input_dim = 39
n_epochs = 300

data_stream = PennActionData(base_dir = '/home/hrishi/1Hrishi/0Thesis/Data/Penn_Action/labels/', file = '0129.mat')
data_len = data_stream.data_len
print(data_len)
sequences = data_stream.get_sequence_dict(seq_length = seq_length)

# LSTM
model = PoseLSTM(input_dim = input_dim, hidden_dim = hidden_dim,
                batch_size = batch_size, output_dim = output_dim, num_layers = num_layers)
# Initialise hidden state
model.hidden = model.init_hidden()
# model.double().cuda()
model.cuda()
loss_fn = nn.MSELoss(reduction = 'elementwise_mean')
optimizer = optim.SGD(model.parameters(), lr = 1e-4)

# try:
average_losses = []
print("Training for {} epochs...".format(n_epochs))
for epoch in tqdm(range(1, n_epochs + 1)):
    # Clear stored gradient
    model.zero_grad()
    loss = 0
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

        ## Sanity Checks :p
        # print("Complete Sequence shape: ", sequence.shape)
        # print("Training Sequence shape: ", sequence[:,:-1].shape)
        # print("Target Sequence shape: ", sequence[:,1:].shape)
        # print("Predicted Sequence shape: ", y_pred.shape)

        # Calculate Loss
        loss = loss + loss_fn(y_pred, sequence[:,1:])
    av_epoch_loss = loss.item()//seq_length
    print("Average loss over {} sequences: {} @ epoch #{}".format(data_len, av_epoch_loss, epoch))
    average_losses.append(av_epoch_loss)
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward(retain_graph = True)
    optimizer.step()
# except:
#     print("Exiting")

# Save the model checkpoint
checkpoint_file = "model_epoch" + str(n_epochs) + "_train_loss" + str(average_losses[-1]) + ".ckpt"
torch.save(model.state_dict(), checkpoint_file)
#
# checkpoint = torch.load('model.ckpt')
# model.load_state_dict(checkpoint)

# Test the model
with torch.no_grad():
    from sklearn.metrics import explained_variance_score, r2_score, mean_squared_error
    # Get random training sample
    rand_key = np.random.randint(low = 0, high = data_len)
    x_test = sequences.get(rand_key, sequences.get(0))
    sequence = torch.from_numpy(x_test).float()
    sequence = sequence.cuda()
    sequence = sequence.view((batch_size, -1, input_dim))
    # Evaluate
    y_pred = model.forward(sequence[:,:-1])
    print(y_pred[0].shape)
    explained_variance = explained_variance_score(x_test[1:], y_pred[0])
    r2 = r2_score(x_test[1:], y_pred[0])
    mse = mean_squared_error(x_test[1:], y_pred[0])
    print("Prediction metrics for {}th training sample:\n".format(rand_key+1))
    print("Explained Variance score: {}\n".format(explained_variance))
    print("R2 score: {}\n".format(r2))
    print("MSE: {}\n".format(mse))

# # Save the model checkpoint
# torch.save(model.state_dict(), 'model.ckpt')

import matplotlib.pyplot as plt
plt.plot(average_losses)
plt.show()
