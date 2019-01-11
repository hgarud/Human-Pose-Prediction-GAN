import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from Data import PennActionData
from losses import DiscriminatorLoss, GeneratorLoss
from Models import PoseRGAN
from utils import mse_measure, r2_measure

device = torch.device('cuda')

# Model Hyper-Parameters
hidden_dim = 128
num_layers = 2
seq_length = 16
num_epochs = 50
alpha_G = 1e-3
alpha_D = 1e-2

with open('action_to_video_train.txt', 'rb') as handle:
    action_to_video_train = pickle.loads(handle.read())
with open('action_to_video_test.txt', 'rb') as handle:
    action_to_video_test = pickle.loads(handle.read())

# Actions = ['baseball_swing', 'bowl', 'jumping_jacks', 'tennis_serve', 'situp',
            # 'squat', 'strum_guitar', 'pushup', 'pullup', 'tennis_forehand',
            # 'bench_press', 'jump_rope', 'baseball_pitch', 'clean_and_jerk', 'golf_swing']

action = 'baseball_swing'
vidoes =
data_stream = PennActionData(base_dir = '/home/hrishi/1Hrishi/0Thesis/Data/Penn_Action/labels/', file = '0758.mat', scaling = 'standard')
data_len = data_stream.data_len
print(data_len)
sequences = data_stream.getStridedSequences(seq_length = seq_length, withVisibility = False)
np.random.shuffle(sequences)
print(sequences.shape)

# sequences = np.load('All_16SL_26F_Sequences_Standard.npy')
# data_len = len(sequences)
# print(sequences.shape)
# np.random.shuffle(sequences)

train_size = 1
split_index = math.ceil(data_len * train_size) # 122869
print(split_index)

train_sequences = sequences[0:split_index]
validation_sequences = sequences[split_index:]

input_dim = sequences.shape[-1]         # 26 when withVisibility = False
                                        # 39 when withVisibility = True

output_dim = input_dim
# batch_size = 6553
batch_size = 647
total_train_steps = len(train_sequences)/batch_size
total_valid_steps = len(validation_sequences)/batch_size

gan = PoseRGAN(input_dim = input_dim, hidden_dim = hidden_dim,
                batch_size = batch_size, output_dim = output_dim, num_layers = num_layers)

# PATH = 'NonVisibilitySTD_LSTMmodel_epoch200_valid_loss0.11372496634721756.ckpt'
netG = gan.netG
# netG.load_state_dict(torch.load(PATH, map_location = 'cpu'))
netG.hidden = netG.init_hidden()
netG.cuda()
netD = gan.netD
netD.hidden = netD.init_hidden()
netD.cuda()

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

G_optimizer = optim.SGD(netG.parameters(), lr = alpha_G)
# D_optimizer = optim.SGD(netD.parameters(), lr = alpha_D)
D_optimizer = optim.Adam(netD.parameters(), lr = alpha_D)
G_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer = G_optimizer, mode = 'min', factor = 0.5, verbose = True)
D_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer = D_optimizer, mode = 'min', factor = 0.5, verbose = True)

gLoss = GeneratorLoss(lambda_A = 0.8, lambda_P = 1, lambda_D = 0)
dLoss = DiscriminatorLoss()

# X_train = torch.from_numpy(sequences[:,:-1,:]).float().to(device)
# X_train = X_train.view((batch_size, -1, input_dim))
#
# y_train = torch.from_numpy(sequences[:,1:,:]).float().to(device)
# y_train = y_train.view((batch_size, -1, input_dim))

 # Training Loop
# Lists to keep track of progress
G_losses = []
D_losses = []
# valid_losses = []
r2s = []
mses = []

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    mean_epoch_G_loss = 0
    mean_epoch_D_loss = 0
    for batch_index in range(0, train_sequences.shape[0], batch_size):
        X_train = torch.from_numpy(train_sequences[batch_index:batch_index + batch_size,:-1,:]).float().to(device)
        X_train = X_train.view((batch_size, -1, input_dim))

        y_train = torch.from_numpy(train_sequences[batch_index:batch_index + batch_size,1:,:]).float().to(device)
        y_train = y_train.view((batch_size, -1, input_dim))

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()

        label = torch.full((batch_size, seq_length-1, input_dim), real_label, device=device)
        # Forward pass real batch through D
        # output = netD(X_train)
        output = netD(y_train)
        # Calculate loss on all-real batch
        errD_real = dLoss(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate fake data batch with G
        fake = netG(X_train).detach()
        # fake = netG(y_train).detach()
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake)
        # Calculate D's loss on the all-fake batch
        errD_fake = dLoss(input = output, target = label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # print("Loss over {} sequences: {} for step {}/{} @ epoch #{}/{}".format(data_len, errD, (batch_index/batch_size) + 1, total_train_steps, epoch, num_epochs))

        # Update D
        D_optimizer.zero_grad()
        D_optimizer.step()


        ############################
        # (2) Update G network: maximize log(D(G(z))) + MSELoss(x, G(z))
        ###########################
        netG.zero_grad()
        G_optimizer.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake)
        # Calculate G's loss based on this output
        errG = gLoss(input = output, target = y_train, target_label = label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        G_optimizer.step()

        # Output training stats
        # if epoch % 5 == 0:
        print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'% (epoch, num_epochs, errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        mean_epoch_G_loss = mean_epoch_G_loss + errG.item()
        mean_epoch_D_loss = mean_epoch_D_loss + errD.item()

    # Save Losses for plotting later
    G_losses.append(mean_epoch_G_loss/total_train_steps)
    D_losses.append(mean_epoch_D_loss/total_train_steps)
    '''
    mean_epoch_valid_loss = 0
    mean_epoch_valid_r2 = 0
    mean_epoch_valid_mse = 0
    for batch_index in range(0, validation_sequences.shape[0], batch_size):
        X_valid = torch.from_numpy(validation_sequences[batch_index:batch_index+batch_size,:-1,:]).float().to(device)
        X_valid = X_valid.view((-1, seq_length - 1, input_dim))

        y_valid = torch.from_numpy(validation_sequences[batch_index:batch_index+batch_size,1:,:]).float().to(device)
        y_valid = y_valid.view((-1, seq_length - 1, input_dim))

        # Forward pass
        y_pred = netG.forward(X_valid.detach())

        # valid_loss = gLoss(input = y_pred, target = y_valid)
        mean_epoch_valid_mse = mean_epoch_valid_mse + mse_measure(y_pred, y_valid)
        mean_epoch_valid_r2 = mean_epoch_valid_r2 + r2_measure(y_pred, y_valid)
        # mean_epoch_valid_loss = mean_epoch_valid_loss+ valid_loss.item()
    # valid_losses.append(mean_epoch_valid_loss/total_valid_steps)
    r2s.append(mean_epoch_valid_r2/total_valid_steps)
    mses.append(mean_epoch_valid_mse/total_valid_steps)

    G_scheduler.step(mses[-1])
    '''
    G_scheduler.step(G_losses[-1])
    D_scheduler.step(D_losses[-1])

# Save the model checkpoint
# G_checkpoint_file = "Visibility_RGANmodel_epoch" + str(epoch) + "_valid_mseloss" + str(mses[-1]) + ".ckpt"
# D_checkpoint_file = "Visibility_RGANmodel_epoch" + str(epoch) + "_valid_mseloss" + str(mses[-1]) + ".ckpt"
# torch.save(netG.state_dict(), G_checkpoint_file)
# torch.save(netD.state_dict(), D_checkpoint_file)

fig = plt.figure()
plt.plot(D_losses, 'k')
plt.plot(G_losses, 'r')

# fig = plt.figure()
# plt.plot(r2s, 'b')
# plt.plot(mses, 'g')
plt.show()
