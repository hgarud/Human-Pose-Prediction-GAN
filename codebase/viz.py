from Data import PennActionData
from Models import PoseLSTM
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from matplotlib.patches import Circle

device = torch.device('cuda')

# Model Hyper-Parameters
hidden_dim = 128
num_layers = 2
seq_length = 16
# n_epochs = 200
# alpha = 1e-2

def visualize(data, scalers, path, count = 5, subset = ''):
    data_len = len(data)
    print(data.shape)
    # index = random.randint(0, data_len - count)
    index = 0
    X = data[index:index+count,:,:]
    print(X.shape)
    for i, sequence in enumerate(X):
        for j, frame in enumerate(sequence):
            image = np.zeros((360, 480))
            x_coord = frame[0:13]
            y_coord = frame[13:26]
            x_coord = scalers[0].inverse_transform(x_coord)
            y_coord = scalers[1].inverse_transform(y_coord)
            fig,ax = plt.subplots(1)
            ax.imshow(image)
            z = zip(x_coord, y_coord)
            for x,y in z:
                circ = Circle((x, y), 5)
                ax.add_patch(circ)
            assert path[-1] == '/'
            image_name = path + subset + "image_{}_{}.png".format(i,j)
            fig.savefig(image_name)

data_stream = PennActionData(base_dir = '/home/hrishi/1Hrishi/0Thesis/Data/Penn_Action/labels/', scaling = 'standard')
scalers = (data_stream.x_scaler, data_stream.y_scaler)
print(scalers[0].scale_)

# Load saved Numpy file
sequences = np.load('All_16SL_26F_Sequences_Standard.npy')
data_len = len(sequences)
print(sequences.shape)
# np.random.shuffle(sequences)

input_dim = sequences.shape[-1]         # 26 when withVisibility = False
                                        # 39 when withVisibility = True

output_dim = input_dim

batch_size = 6553

PATH = 'NonVisibilitySTD_LSTMmodel_epoch200_valid_loss0.11372496634721756.ckpt'

# LSTM
model = PoseLSTM(input_dim = input_dim, hidden_dim = hidden_dim,
                batch_size = batch_size, output_dim = output_dim, num_layers = num_layers)

model.load_state_dict(torch.load(PATH, map_location = 'cpu'))
# Initialise hidden state
model.hidden = model.init_hidden()
model.eval()
model.cuda()

# batch_index = random.randint(0, data_len - batch_size)
batch_index = 0
# for batch_index in range(0, train_sequences.shape[0], batch_size):
X_test = torch.from_numpy(sequences[batch_index:batch_index+batch_size,:-1,:]).float().to(device)
X_test = X_test.view((-1, seq_length - 1, input_dim))

y_test = torch.from_numpy(sequences[batch_index:batch_index+batch_size,1:,:]).float().to(device)
y_test = y_test.view((-1, seq_length - 1, input_dim))

# Clear stored gradient
model.zero_grad()

# Forward pass
y_pred = model.forward(X_test)
# print(y_pred.shape)
# print(type(y_pred))
y_pred = y_pred.detach().data.cpu().numpy()
# print(y_pred.dtype)
# # y_pred =
y_test = y_test.detach().data.cpu().numpy()
visualize(y_pred, subset = 'predicted', scalers = scalers, path = '/home/hrishi/1Hrishi/0Thesis/viz/')
visualize(y_test, subset = 'testing', scalers = scalers, path = '/home/hrishi/1Hrishi/0Thesis/viz/')
# print(type(y_pred))
# print(y_pred.shape)
