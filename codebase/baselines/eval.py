import numpy as np
import matplotlib.pyplot as plt
from Baseline_Models import ERD, LSTM3LR
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn.externals import joblib
from Data import un_normalize_2d
import sys
sys.path.append("../")
import torch

device = torch.device('cuda')

def visualize(data, path, bBox, count=1, subset='', scalers=None):
    data_len = len(data)
    print(data.shape)
    # index = random.randint(0, data_len - count)
    un_normalized_data = un_normalize_2d(data, bBox[:, 0], bBox[:, 2], bBox[:, 1], bBox[:, 3], bBox[:, 4])
    print(un_normalized_data.shape)
    index = 0
    X = un_normalized_data[index:index+count,:,:]
    # print(X.shape)
    for i, sequence in enumerate(X):
        # print("Sequence shape: ", sequence.shape)
        # sequence[:,0:13] = scalers[0].inverse_transform(sequence[:,0:13])
        # sequence[:,13:26] = scalers[1].inverse_transform(sequence[:,13:26])
        for j, frame in enumerate(sequence):
            image = np.zeros((360, 480))
            # print("Frame shape: ", frame.shape)
            x_coord = frame[0:13]
            y_coord = frame[13:26]
            # print(x_coord[None,:].shape)
            # x_coord = scalers[0].inverse_transform(x_coord[None, :])[0]
            # y_coord = scalers[1].inverse_transform(y_coord[None, :])[0]
            # print(x_coord[None,:].shape)
            fig,ax = plt.subplots(1)
            ax.imshow(image)
            z = zip(x_coord, y_coord)
            # print(next(z))
            for x,y in z:
                # print(x,y)
                circ = Circle((x, y), 5)
                ax.add_patch(circ)
            assert path[-1] == '/'
            image_name = path + subset + "image_{}_{}.png".format(i,j)
            fig.savefig(image_name)
            plt.cla()

# def greedy_decode(model, src, max_len, start_frame):

sequences = np.load('../Normalized_Test_16SL_26F_Sequences.npy')
data_len = len(sequences)

bBoxForSequences = np.array(np.load('../Normalizing_Test_BBOX_Params.npy'))

input_dim = sequences.shape[-1]
output_dim = input_dim
batch_size = 5
batch_index = 0
seq_length = sequences.shape[1]
dropout=0.1

PATH = '/home/hrishi/1Hrishi/0Thesis/Human-Pose-Prediction-GAN/codebase/baselines/LSTM_3LR_DO0.1_loss_0.0054000611416995525.ckpt'
# model = ERD(input_dim=input_dim, output_dim=output_dim, batch_size=batch_size, dropout=0.2)
model = LSTM3LR(input_dim=input_dim, hidden_dim=1000, output_dim=output_dim, num_layers=3, batch_size=batch_size, dropout=dropout)
model.hidden = model.init_hidden()
model.load_state_dict(torch.load(PATH, map_location = 'cpu'))
model.eval()
model.cuda()


X_test = torch.from_numpy(sequences[batch_index:batch_index+batch_size,:-1,:]).float().to(device)
# X_test = X_test.view((-1, seq_length - 1, input_dim))

y_test = torch.from_numpy(sequences[batch_index:batch_index+batch_size,1:,:]).float().to(device)
# y_test = y_test.view((-1, seq_length - 1, input_dim))
model.zero_grad()
y_pred = model.forward(X_test)

y_pred = y_pred.detach().data.cpu().numpy()
y_test = y_test.detach().data.cpu().numpy()

bBox = bBoxForSequences[batch_index:batch_index+batch_size]

pck = np.zeros((y_test.shape[0], y_test.shape[1]))
euclidean = np.zeros((y_test.shape[0], y_test.shape[1]))
for i, sequence in enumerate(y_pred):
    for j, frame, in enumerate(sequence):
        diff = frame - y_test[i, j, :]
        joint_distance = np.sqrt(np.square(diff[0:13]) + np.square(diff[13:26]))
        euclidean[i,j] = np.average(joint_distance)
        in_threshold_points = (joint_distance < 0.1)
        pck[i, j] = np.sum(in_threshold_points) / 13

pck = np.mean(pck, axis = 0)
euclidean = np.mean(euclidean, axis = 0)
print("Average PCK per frame for all sequences sequence: ", pck)
print("Average Euclidean distance per frame for all sequences sequence: ", euclidean)
fig = plt.figure()
plt.plot(pck, 'k', label = 'pck')
plt.plot(euclidean, 'r', label = 'euclidean')
plt.legend()
plt.show()

visualize(y_pred, subset = 'predicted', bBox = bBoxForSequences[batch_index:batch_index+batch_size], path = '/home/hrishi/1Hrishi/0Thesis/viz/')
visualize(y_test, subset = 'testing', bBox = bBoxForSequences[batch_index:batch_index+batch_size], path = '/home/hrishi/1Hrishi/0Thesis/viz/')
