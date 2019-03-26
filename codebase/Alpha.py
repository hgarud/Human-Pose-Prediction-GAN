import resnext
import resnet
import argparse
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import cv2
from MyResnet import enc_resnet18, dec_resnet18, getImageBatch
from torchvision import transforms
from DataLoaders import TrainDataset, ToTensor
from sklearn.utils import shuffle
from Attention import MultiHeadedAttention
from BumbleBee import EncoderDecoder, Generator
from Decoder import Decoder, DecoderLayer
from Encoder import Encoder, EncoderLayer
from IronHide import Batch, LabelSmoothing, NoamOpt, SimpleLossCompute
from Model import PositionwiseFeedForward, Embeddings, PositionalEncoding, make_model, data_gen
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader

random_state = np.random.RandomState(seed=5)

def pose_data_loader(sequences, batch_size, index):
    data = torch.from_numpy(sequences[index:index+batch_size, :, :])
    src = Variable(data[:, :8, :].float().cuda(), requires_grad=False)
    tgt = Variable(data[:, 8:, :].float().cuda(), requires_grad=False)
    return Batch(src, tgt)

def get_latent_image_sequence_batch(batch, encoder, latent_dim):
    assert batch['source'].shape[1] == batch['target'].shape[1]
    latent_src = torch.cuda.FloatTensor(batch['source'].shape[0], batch['source'].shape[1], latent_dim).fill_(0)
    latent_trg = torch.cuda.FloatTensor(batch['target'].shape[0], batch['target'].shape[1], latent_dim).fill_(0)
    for t in range(batch['source'].shape[1]):
        src = batch['source'][:,t,:,:,:].float().cuda()
        trg = batch['target'][:,t,:,:,:].float().cuda()
        z_src, mu_src, logvar_src, indices_src = encoder(src)
        z_trg, mu_trg, logvar_trg, indices_trg = encoder(trg)

        latent_src[:,t,:] = z_src
        latent_trg[:,t,:] = z_trg

    return Batch(latent_src, latent_trg)

latent_dim = 5
encoder = enc_resnet18(latent_dim = latent_dim)
# encoder.load_state_dict(torch.load('FrameEncVae.ckpt', map_location = 'cpu'))
encoder.float().cuda()
encoder.train()

frame_sequences = np.load('Preprocessed_Train_16SL_Frame_Sequences_NoBenchPress.npy')
print(frame_sequences.shape)

pose_sequences = np.load('Normalized_Train_16SL_26F_Sequences_NoBenchPress.npy')
print(pose_sequences.shape)
data_len = len(pose_sequences)
frame_sequences, pose_sequences = shuffle(frame_sequences, pose_sequences, random_state = random_state)

batch_size = 4
input_dim = pose_sequences.shape[-1] + latent_dim
output_dim = input_dim
total_train_steps = data_len//batch_size

train_loader =  TrainDataset(
                    base_dir = "/media/hrishi/OS/1Hrishi/1Cheese/0Thesis/Data/Penn_Action/preprocessed/frames/",
                    data = frame_sequences,
                    batch_size = batch_size,
                    shuffle = False,
                    random_state = random_state,
                    transform=transforms.Compose([ToTensor()])
                    )

dataloader = DataLoader(train_loader, batch_size=batch_size,
                    shuffle=True, num_workers=1)

model = make_model(src_vocab=input_dim, tgt_vocab=output_dim, N=2, d_model=512, d_ff=2048, h=8, dropout=0.1)
model.train()
model.float().cuda()

model_opt = NoamOpt(model.src_embed[0].d_model, 1, 3000,
    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

mse_loss_fn = nn.MSELoss()

n_epochs = 2

train_losses = []
print("Training for {} epochs...".format(n_epochs))
for epoch in range(n_epochs):
    for batch_index, image_sequence_batch in enumerate(dataloader):
        model.zero_grad()
        encoder.zero_grad()

        latent_image_sequence_batch = get_latent_image_sequence_batch(image_sequence_batch, encoder, 5)

        pose_sequence_batch = pose_data_loader(pose_sequences, batch_size, index = batch_index)

        source = torch.cat((pose_sequence_batch.src, latent_image_sequence_batch.src), 2)
        target = torch.cat((pose_sequence_batch.trg, latent_image_sequence_batch.trg), 2)
        target_mask = pose_sequence_batch.trg_mask

        out = model.forward(source, target, target_mask)
        out = model.generator(out)
        print(out.shape)

        loss = mse_loss_fn(out[:, :, :26], pose_sequence_batch.trg)
        train_losses.append(loss.item())
        print("Loss over {} sequences: {} for step {}/{} @ epoch #{}/{}".format(data_len,
                                loss, batch_index,
                                total_train_steps, epoch, n_epochs))
        model_opt.optimizer.zero_grad()
        loss.backward()
        model_opt.step()
