import torch
import torch.utils.data
import numpy as np
import cv2
from torchvision import transforms

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, base_dir, data, batch_size, shuffle=True, transform=None):
        self.base_dir = base_dir
        assert base_dir[-1] == "/"
        if shuffle:
            np.random.shuffle(data)
        self.seq_length = data.shape[1]//2
        print(type(self.seq_length))
        self.batch_size = batch_size
        self.source = data[:, 0:self.seq_length]
        self.target = data[:, self.seq_length:(self.seq_length*2)]
        self.transform = transform


    def __len__(self):
        return len(self.source)

    def getBatch(self, data, index):
        images = np.zeros((self.batch_size, self.seq_length, 256, 256, 3))
        batch_sequence = data[index:index+batch_size]
        self.batch_image_names = []
        for i, sequence_paths in enumerate(batch_sequence):
            for j, path in enumerate(sequence_paths):
                # try:
                    dir = path[0:4]
                    path = path[5:]
                    im_path = dir+"/"+path
                    self.batch_image_names.append(im_path)
                    image = cv2.imread(self.base_dir+im_path)
                    # image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
                    image = self.letterbox_image(image, (256, 256))
                    # image = image/255
                    images[i,j,:,:,:] = image
                except Exception as e:
                    print("Exiting with exception: ", e)
                    print(self.base_dir+path)

        return images

    def letterbox_image(self, img, inp_dim):
        '''resize image with unchanged aspect ratio using padding'''
        img_w, img_h = img.shape[1], img.shape[0]
        w, h = inp_dim
        new_w = int(img_w * min(w / img_w, h / img_h))
        new_h = int(img_h * min(w / img_w, h / img_h))
        resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

        canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image

        return canvas


    def __getitem__(self, index):
        source_batch = self.getBatch(self.source, index)
        target_batch = self.getBatch(self.target, index)

        sample = {'source':source_batch, 'target':target_batch, 'image_paths':self.batch_image_names}
        if self.transform:
            sample = self.transform(sample)
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        source, target = sample['source'], sample['target']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        source = source.transpose((0, 1, 4, 2, 3))
        target = target.transpose((0, 1, 4, 2, 3))
        return {'source': torch.from_numpy(source),
                'target': torch.from_numpy(target)}


if __name__ == '__main__':
    sequences = np.load('Training_Frame_Sequences.npy')
    print(sequences.shape)
    batch_size = 5
    train_loader =  TrainDataset(
                        base_dir = "/media/hrishi/OS/1Hrishi/1Cheese/0Thesis/Data/Penn_Action/preprocessed/frames/",
                        data = sequences,
                        batch_size = batch_size,
                        shuffle = True,
                        transform=transforms.Compose([ToTensor()])
                        )

    for i, batch in enumerate(train_loader):
        print(i)
        # src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.cuda(), batch)
        print(len(batch))
        print(batch['source'].shape)
        break
