from MyResnet import enc_resnet18, dec_resnet18, getImageBatch
import numpy as np
import torch
import matplotlib.pyplot as plt

device = torch.device('cuda')

frame_paths = np.load('All_Training_Frames.npy')[0:32]
# np.random.shuffle(frame_paths)
# data_len = len(frame_paths)
# print(data_len)
# total_train_steps = data_len/batch_size
encoder = enc_resnet18()
encoder.load_state_dict(torch.load('FrameEncVae.ckpt', map_location = 'cpu'))
encoder.cuda()
encoder.eval()
# # out, indices = encoder.forward(x)
decoder = dec_resnet18().cuda()
decoder.load_state_dict(torch.load('FrameDecVae.ckpt', map_location = 'cpu'))
decoder.cuda()
decoder.eval()

def viz(X, mode):
    X = X.permute(0, 2, 3, 1).detach().cpu().numpy()
    for i in range(X.shape[0]):
        fig,ax = plt.subplots(1)
        ax.imshow(X[i])
        image_name = "Vae_eval_" + mode + "_" + str(i) + ".png"
        fig.savefig(image_name)



with torch.no_grad():
    X_train = getImageBatch(base_dir = "/media/hrishi/OS/1Hrishi/1Cheese/0Thesis/Data/Penn_Action/preprocessed/frames/",
                    paths=frame_paths, batch_size = 32)
    X_train = torch.from_numpy(X_train).permute(0,3,1,2).float().cuda()
    X_train = X_train.view((-1, 3, 256, 256))

    encoder.zero_grad()
    decoder.zero_grad()

    z, mu, logvar, indices = encoder.forward(X_train)
    recon_batch = decoder.forward(z, indices)
    recon_batch *= 255
    print(recon_batch.shape)
    viz(X_train, mode = "test")
    viz(recon_batch, mode = "recon")
    # for image in recon_batch:
