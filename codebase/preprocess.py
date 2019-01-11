import cv2
import scipy.io as sio
import numpy as np
from os import listdir, mkdir, path
import pickle

for mode in ['train', 'test']:
    frame_dir = '/home/hrishi/1Hrishi/0Thesis/Data/Penn_Action/frames/'
    label_dir = '/home/hrishi/1Hrishi/0Thesis/Data/Penn_Action/labels/'
    out_frame_base_dir = '/media/hrishi/OS/1Hrishi/1Cheese/0Thesis/Data/Penn_Action/preprocessed/frames/'
    out_label_base_dir = '/media/hrishi/OS/1Hrishi/1Cheese/0Thesis/Data/Penn_Action/preprocessed/labels/'
    pad = 5
    with open('action_to_video_'+mode+'.txt', 'rb') as handle:
        f = pickle.loads(handle.read())
#   f = open('./datasets/PennAction/'+mode+'_list.txt','r')
#   lines = f.readlines()
#   f.close()
#   numvids=len(lines)
#
#   for i, line in enumerate(lines):
#     tokens = line.split()[0].split('frames')
    actions = sorted(f.keys())
    for action in actions:
        videos = sorted(f[action])
        for video in videos:
            # print(video)
            ff = sio.loadmat(label_dir + video)
            bboxes = ff['bbox']
            posey = ff['y']
            posex = ff['x']
            visib = ff['visibility']
            # print(frame_dir + video.split('.')[0] + '/')
            imgs = sorted(listdir(frame_dir + video.split('.')[0] + '/'))
            # print(len(imgs))
            # print(imgs[0])
            box = np.zeros((4,), dtype='int32')
            bboxes = bboxes.round().astype('int32')

            if len(imgs) > bboxes.shape[0]:
                bboxes = np.concatenate((bboxes,bboxes[-1][None]),axis=0)

            box[0] = bboxes[:,0].min()
            box[1] = bboxes[:,1].min()
            box[2] = bboxes[:,2].max()
            box[3] = bboxes[:,3].max()

            for j in range(len(imgs)):
                img = cv2.imread(frame_dir + video.split('.')[0] + '/'+imgs[j])
                y1 = box[1] - pad
                y2 = box[3] + pad
                x1 = box[0] - pad
                x2 = box[2] + pad

                h = y2 - y1 + 1
                w = x2 - x1 + 1
                if h > w:
                    left_pad  = (h - w) / 2
                    right_pad = (h - w) / 2 + (h - w)%2

                    x1 = x1 - left_pad
                    if x1 < 0:
                        x1 = 0

                    x2 = x2 + right_pad
                    if x2 > img.shape[1]:
                        x2 = img.shape[1]

                elif w > h:
                    up_pad = (w - h) / 2
                    down_pad = (w - h) / 2 + (w - h) % 2

                    y1 = y1 - up_pad
                    if y1 < 0:
                        y1 = 0

                    y2 = y2 + down_pad
                    if y2 > img.shape[0]:
                        y2 = img.shape[0]

                cvisib = visib[j]
                if y1 >= 0:
                    cposey = posey[j] - y1
                else:
                    cposey = posey[j] - box[1]

                if x1 >= 0:
                    cposex = posex[j] - x1
                else:
                    cposex = posex[j] - box[0]

                if y1 < 0:
                    y1 = 0
                if x1 < 0:
                    x1 = 0

                patch = img[int(y1):int(y2),int(x1):int(x2)]
                bboxes[j] = np.array([x1, y1, x2, y2])
                posey[j] = cposey
                posex[j] = cposex
                if not path.isdir(out_frame_base_dir + video.split('.')[0]):
                    mkdir(out_frame_base_dir + video.split('.')[0])
                cv2.imwrite(out_frame_base_dir + video.split('.')[0] + '/' + imgs[j].split('.')[0] + '_cropped.png', patch)

                ff['bbox'] = bboxes
                ff['y'] = posey
                ff['x'] = posex
                np.savez(out_label_base_dir + video.split('.')[0] + '.npz', **ff)
                # print(str(i)+'/'+str(numvids)+' '+mode+' processed')
