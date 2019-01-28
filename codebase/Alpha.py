import resnext
import resnet
import argparse
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
# from PIL import Image
import cv2
from MyResnet import enc_resnet18, dec_resnet18, getImageBatch
from StackedHourGlass import StackedHourGlass


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
# encoder = enc_resnet18()
# encoder.load_state_dict(torch.load('FrameEncVae.ckpt', map_location = 'cpu'))
# encoder.cuda()
# encoder.eval()
# decoder = dec_resnet18().cuda()
# decoder.load_state_dict(torch.load('FrameDecVae.ckpt', map_location = 'cpu'))
# decoder.cuda()
# decoder.eval()

PATH = '/home/hrishi/1Hrishi/0Thesis/existing_codes/trainedModels/simpleHG.pth'
model = StackedHourGlass(nChannels=256, nStack=2, nModules=2, numReductions=4, nJoints=16)
model.load_state_dict(torch.load(PATH, map_location = 'cpu')['model_state'])
model.cuda()
model.eval()






#
# # model = resnext.resnet101(shortcut_type=opt.resnet_shortcut, cardinality=opt.resnext_cardinality,
# #                          sample_size=opt.sample_size, sample_duration=opt.sample_duration,
# #                          last_fc=last_fc)
#
# # original saved file with DataParallel
# state_dict = torch.load(opt.model)['state_dict']
# # create new OrderedDict that does not contain `module.`
# from collections import OrderedDict
# new_state_dict = OrderedDict()
# for k, v in state_dict.items():
#     name = k[7:] # remove `module.`
#     new_state_dict[name] = v
#
# # print(new_state_dict)
# model_data = torch.load(opt.model)
# model.load_state_dict(new_state_dict)
# # model.cuda()
# model.eval()
# if opt.verbose:
#     print(model)



# frames = np.load('Training_Frame_Sequences.npy')
# print(frames.shape)
# image =
# #
#
# x = torch.rand(1, 3, 1, 112, 112)
# print("We are starting with:", x.shape)
#
# out = model.forward(x)
# print(out.shape)

with torch.no_grad():
    X_train = cv2.imread('/media/hrishi/OS/1Hrishi/1Cheese/0Thesis/Data/Penn_Action/preprocessed/frames/0001/000001_cropped.png')
    X_train = cv2.resize(X_train, (256, 256))
    X_train = torch.from_numpy(X_train).permute(2,0,1).float().cuda()
    X_train = X_train.view((-1, 3, 256, 256))
#
    out = model.forward(X_train)
    print(type(out))
    print(len(out))
    print(len(out[0]))
    print(type(out[0]))
    out = out[1]
    print(out.shape)
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
