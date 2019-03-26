import torch
import torch.utils.data
import numpy as np
import cv2
from torchvision import transforms

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, base_dir, data, batch_size, shuffle=True, transform=None, random_state=None):
        self.base_dir = base_dir
        assert base_dir[-1] == "/"
        if shuffle:
            assert random_state
            random_state.shuffle(data)
        self.seq_length = data.shape[1]//2
        self.batch_size = batch_size
        self.source = data[:, 0:self.seq_length]
        self.target = data[:, self.seq_length:(self.seq_length*2)]
        self.transform = transform


    def __len__(self):
        return len(self.source)

    def getBatch(self, data, index):
        images = np.zeros((self.batch_size, self.seq_length, 256, 256, 3))
        ratios = np.zeros((self.batch_size))
        batch_sequence = data[index:index+self.batch_size]
        for i, sequence_paths in enumerate(batch_sequence):
            for j, path in enumerate(sequence_paths):
                try:
                    dir = path[0:4]
                    path = path[5:]
                    im_path = dir+"/"+path
                    image = cv2.imread(self.base_dir+im_path)
                    image, ratio = self.letterbox_image(image, 256)
                    # image = image/255
                    images[i,j,:,:,:] = image
                    ratios[i] = ratio
                except Exception as e:
                    print("Exiting with exception: ", e)
                    print(self.base_dir+path)

        return images, ratios

    def getSequence(self, data, index):
        images = np.zeros((self.seq_length, 256, 256, 3))
        ratio = 0
        sequence_paths = data[index]
        for j, path in enumerate(sequence_paths):
            try:
                dir = path[0:4]
                path = path[5:]
                im_path = dir+"/"+path
                image = cv2.imread(self.base_dir+im_path)
                image, ratio = self.letterbox_image(image, 256)
                # image = image/255
                images[j,:,:,:] = image
                ratio = ratio
            except Exception as e:
                print("Exiting with exception: ", e)
                print(self.base_dir+path)

        return images, ratio


    def letterbox_image(self, image, desired_size):
        '''resize image with unchanged aspect ratio using padding'''

        old_size = image.shape[:2]
        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])

        image = cv2.resize(image, (new_size[1], new_size[0]))

        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]

        # top, bottom = delta_h//2, delta_h-(delta_h//2)
        # left, right = delta_w//2, delta_w-(delta_w//2)
        top, bottom = 0, delta_h
        left, right = 0, delta_w

        color = [0, 0, 0]
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=color)
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        return image, ratio

    def __getitem__(self, index):
        # source_batch, source_ratios = self.getBatch(self.source, index)
        # target_batch, target_ratios = self.getBatch(self.target, index)

        source_sequence, source_ratio = self.getSequence(self.source, index)
        target_sequence, target_ratio = self.getSequence(self.target, index)
        # sample = {'source':source_batch, 'target':target_batch, 'image_paths':self.batch_image_names}
        sample = {'source':source_sequence, 'target':target_sequence}
        if self.transform:
            source_sequence, target_sequence = self.transform(sample)

        sample = {'source':source_sequence, 'target':target_sequence, 'source_ratios':source_ratio,
                    'target_ratios':target_ratio}
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        source, target = sample['source'], sample['target']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        source = source.transpose((0, 3, 1, 2))
        target = target.transpose((0, 3, 1, 2))
        return torch.from_numpy(source), torch.from_numpy(target)


if __name__ == '__main__':
    sequences = np.load('Training_Frame_Sequences.npy')
    print(sequences.shape)
    batch_size = 5
    train_loader =  TrainDataset(
                        base_dir = "/media/hrishi/OS/1Hrishi/1Cheese/0Thesis/Data/Penn_Action/preprocessed/frames/",
                        data = sequences,
                        batch_size = batch_size,
                        shuffle = False,
                        transform=transforms.Compose([ToTensor()])
                        )


    dataloader = torch.utils.data.DataLoader(train_loader, batch_size=batch_size,
                        shuffle=True, num_workers=1)

    for i, batch in enumerate(dataloader):
        print(i)
        print(len(batch))
        print(batch['source'].shape)
        break
