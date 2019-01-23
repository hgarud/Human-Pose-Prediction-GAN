import torch
import torch.nn as nn
from torch import optim
import math
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt

def conv3x3(in_planes, out_planes, stride=1):
    # 3x3 convolution with padding
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)

class EncoderBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(EncoderBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class EncoderResNet(nn.Module):

    def __init__(self, block, layers, sample_size=0, sample_duration=0):

        self.inplanes = 64
        super(EncoderResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=(2, 2),
                               padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, return_indices=True)
        self.layer1 = self._make_encoder_layer(block, 64, layers[0])
        self.layer2 = self._make_encoder_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_encoder_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_encoder_layer(block, 512, layers[3], stride=2)
        # last_duration = math.ceil(sample_duration / 16)
        # last_size = math.ceil(sample_size / 32)
        self.maxgpool2 = nn.MaxPool2d(kernel_size=(4, 4), stride=1, return_indices=True)
        # self.fc = nn.Linear(, num_classes)

        self.fc1 = nn.Linear(12800, 2)
        self.fc2 = nn.Linear(12800, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_encoder_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x, mpindx1 = self.maxpool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x, mpindx2 = self.maxgpool2(x)
        pool_indices = (mpindx1, mpindx2)

        x = x.view(x.size(0), -1)

        mu = self.fc1(x)
        logvar = self.fc2(x)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar, pool_indices

class DecoderBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, kernel_size=1):
        super(DecoderBasicBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(inplanes, planes, kernel_size=kernel_size, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

class DecoderResNet(nn.Module):

    def __init__(self, block, layers, sample_size=0, sample_duration=0):
        # self.inplanes = 512
        # self.indices = indices
        super(DecoderResNet, self).__init__()

        self.fc1 = nn.Linear(2, 12800)

        self.maxunpooling1 = nn.MaxUnpool2d(kernel_size=4, stride=1)
        self.layer1 = self._make_decoder_layer(block, 512, 256, layers[0], stride=2)
        self.layer2 = self._make_decoder_layer(block, 256, 128, layers[0], stride=2)
        self.layer3 = self._make_decoder_layer(block, 128, 64, layers[0], stride=2)
        self.layer4 = self._make_decoder_layer(block, 64, 64, layers[0])
        self.maxunpooling2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv = nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def _make_decoder_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.ConvTranspose2d(inplanes, planes, kernel_size=2, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, kernel_size=stride))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, indices):
        x = self.fc1(x)
        x = x.view(-1, 512, 5, 5)

        x = self.maxunpooling1(x, indices = indices[1])
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.maxunpooling2(x, indices = indices[0])
        x = self.conv(x)
        x = self.relu(x)
        return x

def enc_resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = EncoderResNet(EncoderBasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def dec_resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = DecoderResNet(DecoderBasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def getImageBatch(base_dir, paths, batch_size):
    assert base_dir[-1] == "/"
    images = np.zeros((batch_size, 256, 256, 3))
    for i, path in enumerate(paths):
        try:
            # print(base_dir+path)
            image = cv2.imread(base_dir+path)
            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
            image = image/255
            images[i,:,:] = image
        except Exception as e:
            print("Exiting with exception: ", e)
            print(path)

    return images
if __name__ == '__main__':

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(recon_x, x, mu, logvar):
        # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 26), reduction='elementwise_mean')
        BCE = F.mse_loss(recon_x, x.view((-1, 3, 256, 256)), reduction='elementwise_mean')
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

    batch_size = 32
    n_epochs = 2

    frame_paths = np.load('All_Training_Frames.npy')[0:87520]
    np.random.shuffle(frame_paths)
    data_len = len(frame_paths)
    print(data_len)
    total_train_steps = data_len/batch_size
    encoder = enc_resnet18().cuda()
    encoder.train()
    # # out, indices = encoder.forward(x)
    decoder = dec_resnet18().cuda()
    decoder.train()

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer, mode = 'min', factor = 0.5, patience = 50, verbose = True)
    # out = decoder.forward(out)
    # x = torch.rand(32, 3, 256, 256).cuda()
    # print(out.shape)
    # print(indices)
    # print(out.shape)



    train_losses = []
    try:
        print("Training for {} epochs...".format(n_epochs))
        for epoch in range(1, n_epochs + 1):
            mean_epoch_train_loss = 0
            for batch_index in range(0, data_len, batch_size):
                X_train = getImageBatch(base_dir = "/media/hrishi/OS/1Hrishi/1Cheese/0Thesis/Data/Penn_Action/preprocessed/frames/",
                                paths=frame_paths[batch_index:batch_index+batch_size], batch_size = batch_size)
                # print(X_train.shape)
                # break
                X_train = torch.from_numpy(X_train).permute(0,3,1,2).float().cuda()
                # print(X_train.shape)
                X_train = X_train.view((-1, 3, 256, 256))

                encoder.zero_grad()
                decoder.zero_grad()

                z, mu, logvar, indices = encoder.forward(X_train)
                recon_batch = decoder.forward(z, indices)

                loss = loss_function(recon_batch, X_train, mu, logvar)
                print("Loss over {} data points: {} @ epoch #{}/{}".format(data_len, loss, epoch, n_epochs))
                mean_epoch_train_loss = mean_epoch_train_loss+ loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step(loss)
            train_losses.append(mean_epoch_train_loss/total_train_steps)
            scheduler.step(train_losses[-1])
    except Exception as e:
        print("Exiting with exception: ", e)
    # print()

    fig = plt.figure()
    plt.plot(train_losses, 'k')
    plt.show()

    checkpoint_file = "FrameEncVae.ckpt"
    torch.save(encoder.state_dict(), checkpoint_file)
    checkpoint_file = "FrameDecVae.ckpt"
    torch.save(decoder.state_dict(), checkpoint_file)
