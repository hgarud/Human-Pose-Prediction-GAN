import copy
import math
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from Attention import MultiHeadedAttention
from BumbleBee import EncoderDecoder, Generator
from Decoder import Decoder, DecoderLayer
from Encoder import Encoder, EncoderLayer
from IronHide import Batch, LabelSmoothing, NoamOpt, SimpleLossCompute


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    ''' https://stackoverflow.com/questions/47205762/embedding-3d-data-in-pytorch '''
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        # self.lut = nn.Embedding(vocab, d_model)           # "Failed to allocate resource" error. WIP
        self.lut = nn.Linear(vocab, d_model)                # Works for now
        self.d_model = d_model

    def forward(self, x):
        emb = self.lut(x)                                   # b_s, seq_length, d_model
        sqrt = math.sqrt(self.d_model)
        emb = torch.mul(emb, sqrt)
        return emb


class PositionalEncoding(nn.Module):
    """Implement the PE function."""
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * - (math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.0):
    """Helper: Construct a model from hyperparameters."""
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def data_gen(sequences, batch_size, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        # data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data = torch.from_numpy(sequences[0:batch_size, :, :])
        # data[:, 0] = 1
        src = Variable(data[:, :-1, :].cuda(), requires_grad=False)
        tgt = Variable(data[:, 1:, :].cuda(), requires_grad=False)
        yield Batch(src, tgt)


if __name__ == '__main__':

    # Load saved Numpy file
    sequences = np.load('/media/hrishi/OS/1Hrishi/1Cheese/0Thesis/Data/Preprocessed_All_16SL_26F_Sequences.npy')
    data_len = len(sequences)
    print(sequences.shape)
    np.random.shuffle(sequences)

    input_dim = sequences.shape[-1]
    output_dim = input_dim
    batch_size = 97
    total_train_steps = data_len//batch_size

    model = make_model(src_vocab=input_dim, tgt_vocab=output_dim, N=2)
    model.train()
    model.double().cuda()

    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 4000,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    criterion = LabelSmoothing(size=output_dim, smoothing=0.0)
    criterion.cuda()

    loss_compute = SimpleLossCompute(model.generator, criterion, model_opt)
    mse_loss_fn = nn.MSELoss()
    # train_loader =  torch.utils.data.DataLoader(TrainDataset(
    #                     base_dir = "/media/hrishi/OS/1Hrishi/1Cheese/0Thesis/Data/Penn_Action/preprocessed/frames/",
    #                     data = sequences,
    #                     batch_size = batch_size,
    #                     shuffle = True))


    n_epochs = 5

    losses = []
    for epoch in range(n_epochs):
        for batch_index, batch in enumerate(data_gen(sequences, batch_size=batch_size, nbatches=total_train_steps)):
            # print(batch.src.shape)
            model.zero_grad()
            out = model.forward(batch.src, batch.trg, batch.trg_mask)
            # print("#######################################", out.shape)
            out = model.generator(out)
            loss = mse_loss_fn(out, batch.trg)
            losses.append(loss.item())
            print("Loss over {} sequences: {} for step {}/{} @ epoch #{}/{}".format(data_len,
                                    loss, batch_index,
                                    total_train_steps, epoch, n_epochs))
            model_opt.optimizer.zero_grad()
            loss.backward()
            model_opt.step()

    fig = plt.figure()
    plt.plot(losses, 'k')
    plt.show()

    checkpoint_file = "Transformer_loss_" + str(losses[-1]) + ".ckpt"
    torch.save(model.state_dict(), checkpoint_file)
