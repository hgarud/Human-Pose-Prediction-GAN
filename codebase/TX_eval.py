from Model import make_model
from BumbleBee import subsequent_mask
import numpy as np
import matplotlib.pyplot as plt
# import random
from matplotlib.patches import Circle
from sklearn.externals import joblib
import torch
from torch.autograd import Variable
# from viz import visualize
from Data import un_normalize_2d

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

def greedy_decode(model, src, max_len, start_frame, src_mask=None):
    memory = model.encode(src, src_mask)
    # ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    ys = start_frame.view(src.shape[0], -1, src.shape[2])
    # print("Target shape: ", ys.shape)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        out = model.generator(out[:, -1, :]).unsqueeze(1)
        # print("Out shape: ", out.shape)
        # _, next_word = torch.max(prob, dim = 1)
        # next_word = next_word.data[0]
        ys = torch.cat([ys, out], dim=1)
        # print("Concatenation output: ", ys.shape)
                        # torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


# Load saved Numpy file
sequences = np.load('/home/hrishi/1Hrishi/0Thesis/Human-Pose-Prediction-GAN/codebase/Normalized_Test_16SL_26F_Sequences.npy')
data_len = len(sequences)

bBoxForSequences = np.array(np.load('/home/hrishi/1Hrishi/0Thesis/Human-Pose-Prediction-GAN/codebase/Normalizing_Test_BBOX_Params.npy'))

print(bBoxForSequences.shape)
print(sequences.shape)

input_dim = sequences.shape[-1]
output_dim = input_dim

PATH = '/media/hrishi/OS/1Hrishi/1Cheese/0Thesis/checkpoints/BIG_WU2900_Transformer_loss_0.00044433000110201377.ckpt'
model = make_model(src_vocab=input_dim, tgt_vocab=output_dim, N=6, d_model=1024, d_ff=4096, h=16, dropout=0.3)
model.load_state_dict(torch.load(PATH, map_location = 'cpu'))
model.eval()
model.cuda()

batch_size = 5
batch_index = 0
seq_length = 16

# for batch_index in range(0, train_sequences.shape[0], batch_size):
X_test = torch.from_numpy(sequences[batch_index:batch_index+batch_size,:-1,:]).float().to(device)
# X_test = X_test.view((-1, seq_length - 1, input_dim))

y_test = torch.from_numpy(sequences[batch_index:batch_index+batch_size,1:,:]).float().to(device)
# y_test = y_test.view((-1, seq_length - 1, input_dim))
model.zero_grad()

# Forward pass
# y_pred = model.forward(X_test)
y_pred = greedy_decode(model=model, src=X_test, max_len=(y_test.shape[1]), start_frame=y_test[:, 0, :])
print(y_pred.shape)
# print(type(y_pred))
y_pred = y_pred.detach().data.cpu().numpy()
# print(y_pred[:, 1, :])
# print(np.max(y_pred))
# print(np.min(y_pred))
# # y_pred =
y_test = y_test.detach().data.cpu().numpy()
# print(y_test)

bBox = bBoxForSequences[batch_index:batch_index+batch_size]
# y_test = un_normalize_2d(y_test, bBox[:, 0], bBox[:, 2], bBox[:, 1], bBox[:, 3], bBox[:, 4])
# y_pred = un_normalize_2d(y_pred, bBox[:, 0], bBox[:, 2], bBox[:, 1], bBox[:, 3], bBox[:, 4])

print(y_pred.shape)


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

# visualize(y_pred, subset = 'predicted', bBox = bBoxForSequences[batch_index:batch_index+batch_size], path = '/home/hrishi/1Hrishi/0Thesis/viz/')
# visualize(y_test, subset = 'testing', bBox = bBoxForSequences[batch_index:batch_index+batch_size], path = '/home/hrishi/1Hrishi/0Thesis/viz/')
