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
# parser = argparse.ArgumentParser()
# # parser.add_argument('--input', default='input', type=str, help='Input file path')
# # parser.add_argument('--video_root', default='', type=str, help='Root path of input videos')
# parser.add_argument('--model', type=str, help='Model file path', required=True)
# # parser.add_argument('--output', default='output.json', type=str, help='Output file path')
# # parser.add_argument('--mode', default='feature', type=str, help='Mode (score | feature). score outputs class scores. feature outputs features (after global average pooling).')
# parser.add_argument('--batch_size', default=32, type=int, help='Batch Size')
# # parser.add_argument('--n_threads', default=4, type=int, help='Number of threads for multi-thread loading')
# # parser.add_argument('--model_name', default='resnet', type=str, help='Currently only support resnet')
# # parser.add_argument('--model_depth', default=34, type=int, help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
# parser.add_argument('--resnet_shortcut', default='A', type=str, help='Shortcut type of resnet (A | B)')
# # parser.add_argument('--wide_resnet_k', default=2, type=int, help='Wide resnet k')
# parser.add_argument('--resnext_cardinality', default=32, type=int, help='ResNeXt cardinality')
# parser.add_argument('--no_cuda', action='store_true', help='If true, cuda is not used.')
# parser.set_defaults(verbose=False)
# parser.add_argument('--verbose', action='store_true', help='')
# parser.set_defaults(verbose=False)
#
# opt = parser.parse_args()
# opt.sample_duration = 16
# last_fc = True
# opt.sample_size = 32
# #

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
        # print(X_images.shape)
        z_src, mu_src, logvar_src, indices_src = encoder(src)
        z_trg, mu_trg, logvar_trg, indices_trg = encoder(trg)

        latent_src[:,t,:] = z_src
        latent_trg[:,t,:] = z_trg

        # print(z_src.shape)                                  # torch.Size([5, 5])
        # print(mu_src.shape)                                 # torch.Size([5, 5])
        # print(logvar_src.shape)                             # torch.Size([5, 5])
        # print(indices_src[0].shape, indices_src[1].shape)       # torch.Size([5, 64, 64, 64]), torch.Size([5, 512, 5, 5])
        # print(latent_src)
        # break
    return Batch(latent_src, latent_trg)
# for t in range(image_sequence_batch['source'].shape[1]):
#     X_images = image_sequence_batch['source'][:,t,:,:,:].float().cuda()
#     print(X_images.shape)
#     z, mu, logvar, indices = encoder(X_images)
#     print(z.shape)
#     break
latent_dim = 5
encoder = enc_resnet18(latent_dim = latent_dim)
# encoder.load_state_dict(torch.load('FrameEncVae.ckpt', map_location = 'cpu'))
encoder.float().cuda()
encoder.train()
# decoder = dec_resnet18(latent_dim = 2)
# decoder.load_state_dict(torch.load('FrameDecVae.ckpt', map_location = 'cpu'))
# decoder.train()
# decoder.cuda()


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
# valid_losses = []
print("Training for {} epochs...".format(n_epochs))
for epoch in range(n_epochs):
    for batch_index, image_sequence_batch in enumerate(dataloader):
        # print(i)
        model.zero_grad()
        encoder.zero_grad()
        # decoder.zero_grad()
        # src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.cuda(), batch)
        # print(len(image_sequence_batch))
        # print()

        latent_image_sequence_batch = get_latent_image_sequence_batch(image_sequence_batch, encoder, 5)

        pose_sequence_batch = pose_data_loader(pose_sequences, batch_size, index = batch_index)

        source = torch.cat((pose_sequence_batch.src, latent_image_sequence_batch.src), 2)
        target = torch.cat((pose_sequence_batch.trg, latent_image_sequence_batch.trg), 2)
        target_mask = pose_sequence_batch.trg_mask

        out = model.forward(source, target, target_mask)
        # print("#######################################", out.shape)
        out = model.generator(out)
        print(out.shape)

        # break
        loss = mse_loss_fn(out[:, :, :26], pose_sequence_batch.trg)
        train_losses.append(loss.item())
        print("Loss over {} sequences: {} for step {}/{} @ epoch #{}/{}".format(data_len,
                                loss, batch_index,
                                total_train_steps, epoch, n_epochs))
        model_opt.optimizer.zero_grad()
        loss.backward()
        model_opt.step()

# for valid_batch_index, valid_batch in enumerate(data_gen(validation_sequences, batch_size=batch_size, nbatches=total_valid_steps)):
#     model.eval()
#     out = model.forward(valid_batch.src, valid_batch.trg, valid_batch.trg_mask)
#     out = model.generator(out)
#     loss = mse_loss_fn(out, valid_batch.trg)
#     valid_losses.append(loss.item())
# fig = plt.figure()
# # plt.plot(valid_losses, 'k', label = 'validation loss')
# plt.plot(train_losses, 'r', label = 'training loss')
# plt.show()
#
# checkpoint_file = "Transformer_loss_" + str(train_losses[-1]) + ".ckpt"
# torch.save(model.state_dict(), checkpoint_file)




    # print(pose_sequence_batch.src.shape)

# print(frames.shape)
# image =
# #
#
# x = torch.rand(1, 3, 1, 112, 112)
# print("We are starting with:", x.shape)
#
# out = model.forward(x)
# print(out.shape)

# with torch.no_grad():
#     X_train = cv2.imread('/media/hrishi/OS/1Hrishi/1Cheese/0Thesis/Data/Penn_Action/preprocessed/frames/0001/000001_cropped.png')
#     X_train = cv2.resize(X_train, (256, 256))
#     X_train = torch.from_numpy(X_train).permute(2,0,1).float().cuda()
#     X_train = X_train.view((-1, 3, 256, 256))
# #
#     out = model.forward(X_train)
#     print(type(out))
#     print(len(out))
#     print(len(out[0]))
#     print(type(out[0]))
#     out = out[1]
#     print(out.shape)
    # print(len(out[0][0]))
    # print(len(out[0][0][0]))
    # print(len(out[0][0][0][0]))
    # print(len(out[0][0][0][0][0]))
    # print(np.array(out).shape)
    # out = out.permut
#     encoder.zero_grad()
#     decoder.zero_grad()
#
#     z, mu, logvar, indices = encoder.forward(X_train)
#     recon_batch = decoder.forward(z, indices)
#     print(recon_batch.shape)
#     recon_batch = recon_batch.permute(0,2,3,1).detach().cpu().numpy()
#     print(recon_batch.shape)
#     # # fig,ax = plt.subplots(1)
#     # # ax.imshow(recon_batch)
#     # # plt.show()
#     #
    # fig,ax = plt.subplots(1)
    # ax.imshow(out)
    # plt.show()
#     # image_name = "Vae_eval_" + mode + "_" + str(i) + ".png"
#     fig.savefig("image_name.png")
#     # cv2.imshow('image', recon_batch[0])
#     # cv2.waitKey(0)
