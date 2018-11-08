from Data import PennActionData
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch
from Models import PoseRGAN
from losses import GeneratorLoss, DiscriminatorLoss

device = torch.device('cuda')

# Model Hyper-Parameters
hidden_dim = 128
num_layers = 2
seq_length = 16
num_epochs = 500
alpha_G = 1e-3
alpha_D = 1e-2

data_stream = PennActionData(base_dir = '/home/hrishi/1Hrishi/0Thesis/Data/Penn_Action/labels/', file = '0129.mat', scaling = 'minmax')
data_len = data_stream.data_len
print(data_len)
sequences = data_stream.getStridedSequences(seq_length = seq_length, withVisibility = False)
np.random.shuffle(sequences)

input_dim = sequences.shape[-1]         # 26 when withVisibility = False
                                        # 39 when withVisibility = True

output_dim = input_dim
batch_size = sequences.shape[0]

gan = PoseRGAN(input_dim = input_dim, hidden_dim = hidden_dim,
                batch_size = batch_size, output_dim = output_dim, num_layers = num_layers)

netG = gan.netG
netG.hidden = netG.init_hidden()
netG.cuda()
netD = gan.netD
netD.hidden = netD.init_hidden()
netD.cuda()

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

G_optimizer = optim.SGD(netG.parameters(), lr = alpha_G)
D_optimizer = optim.SGD(netD.parameters(), lr = alpha_D)

gLoss = GeneratorLoss(lambda_A = 1, lambda_P = 1, lambda_D = 0)
dLoss = DiscriminatorLoss()

X_train = torch.from_numpy(sequences[:,:-1,:]).float().to(device)
X_train = X_train.view((batch_size, -1, input_dim))

y_train = torch.from_numpy(sequences[:,1:,:]).float().to(device)
y_train = y_train.view((batch_size, -1, input_dim))

 # Training Loop
# Lists to keep track of progress
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):

    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    ## Train with all-real batch
    netD.zero_grad()
    label = torch.full((batch_size, seq_length-1, input_dim), real_label, device=device)
    # Forward pass real batch through D
    output = netD(X_train)
    # Calculate loss on all-real batch
    errD_real = dLoss(output, label)
    # Calculate gradients for D in backward pass
    errD_real.backward()
    D_x = output.mean().item()

    ## Train with all-fake batch
    # Generate fake data batch with G
    fake = netG(X_train)
    label.fill_(fake_label)
    # Classify all fake batch with D
    output = netD(fake.detach())
    # Calculate D's loss on the all-fake batch
    errD_fake = dLoss(input = output, target = label)
    # Calculate the gradients for this batch
    errD_fake.backward()
    D_G_z1 = output.mean().item()
    # Add the gradients from the all-real and all-fake batches
    errD = errD_real + errD_fake
    # Update D
    D_optimizer.step()

    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    netG.zero_grad()
    label.fill_(real_label)  # fake labels are real for generator cost
    # Since we just updated D, perform another forward pass of all-fake batch through D
    output = netD(fake)
    # Calculate G's loss based on this output
    errG = gLoss(training_input = X_train, input = output, target = label)
    # Calculate gradients for G
    errG.backward()
    D_G_z2 = output.mean().item()
    # Update G
    G_optimizer.step()

    # Output training stats
    if epoch % 5 == 0:
        print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
              % (epoch, num_epochs, errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

    # Save Losses for plotting later
    G_losses.append(errG.item())
    D_losses.append(errD.item())
