from Data import PennActionData
import numpy as np
import torch.nn
import torch.optim

"""
Set baselines for the prediction task
* LSTM
* HMM-GMM
* R-GAN
"""

data_stream = PennActionData(base_dir = '/home/hrishi/1Hrishi/0Thesis/Data/Penn_Action/labels/', file = '0758.mat')
data_len = data_stream.data_len

# Model Hyper-Parameters
hidden_dim = 26
batch_size = 663
output_dim = 1
num_layers = 1
seq_length = 16

n_epochs = 5

# LSTM
model = PoseLSTM(input_dim = X_train.shape[-1], hidden_dim = hidden_dim,
                batch_size = batch_size, output_dim = output_dim, num_layers = num_layers)

model.cuda()
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 1e-5)

try:
    print("Training for {} epochs...".format(n_epochs))
    for epoch in tqdm(range(1, n_epochs + 1)):
        # Clear stored gradient
        model.zero_grad()

        # Initialise hidden state
        model.hidden = model.init_hidden()
        model.hidden.cuda()

        for _ in range(data_len//batch_size):
            # Get Random Training Sequence
            X_train, y_train = data_stream.get_random_training_set(seq_len = seq_length, batch_size = batch_size)
            # X_train.cuda()
            # y_train.cuda()

            # Forward pass
            y_pred = model.forward(X_train)

            # Calculate Loss
            loss = loss_fn(y_pred, y_train)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()