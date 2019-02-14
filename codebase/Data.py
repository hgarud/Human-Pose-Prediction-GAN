import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.io import loadmat
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.externals import joblib

def un_normalize(coord, coord_min, coord_max, bounding_box_length):
    coord_mid = coord_min + (coord_max - coord_min) / 2.0
    for i in range(coord.shape[0]):
        coord[i,:,:] = coord[i,:,:] * (bounding_box_length[i] / 2.0) + coord_mid[i]

    return coord

def un_normalize_2d(data, x_min, x_max, y_min, y_max, bounding_box_length):
    new_data = data.copy()
    new_data[:, :, 0:13] = un_normalize(data[:, :, 0:13], x_min, x_max, bounding_box_length)
    new_data[:, :, 13:26] = un_normalize(data[:, :, 13:26], y_min, y_max, bounding_box_length)
    return new_data

class MPIIData(object):
    def __init__(self, base_dir):
        self.base_dir = base_dir
        assert self.base_dir[-1] == '/'

    def load(self, file):
        file = self.base_dir + file
        x = loadmat(file)
        return x

class PennActionData(object):
    def __init__(self, base_dir, is_train, file=None, scaling=None):
        """
        Penn Action Dataset contains 2326 video sequences of 15 different actions
        and human joint annotation for each sequence.

        Annotations for each sequence including class label, coarse viewpoint, human
        body joints, 2D bounding boxes, and training/testing label are contained in separate mat files.

        An example annotation looks like the following in MATLAB:

        annotation =

                action: 'tennis_serve'
                  pose: 'back'
                     x: [46x13 double]
                     y: [46x13 double]
            visibility: [46x13 logical]
                 train: 1
                  bbox: [46x4 double]
            dimensions: [272 481 46]
               nframes: 46

       Reference:
           Weiyu Zhang, Menglong Zhu, Kosta Derpanis,  "From Actemes to Action:
           A Strongly-supervised Representation for Detailed Action Understanding"
           International Conference on Computer Vision (ICCV). Dec 2013.

        Args:
            base_dir (string): Base location where Penn Action labels.
            file (string):     Label MATLAB file to read.
            scaling (string):  Preferred scaling method for coordinates.
                                None:           No scaling
                                'standard':     Standard scaling
                                'minmax':       Min-Max scaling
        """
        self.base_dir = base_dir
        assert self.base_dir[-1] == '/'
        self.video_lengths = []
        self.bBoxParams = []
        if file == None:
            self.data = self.__loadAll(is_train, scaling)
            self.data_len = len(self.data['x'])
        else:
            self.file = file
            self.data = self.__load(file, scaling)
            self.data_len = self.data['nframes'][0,0]

    def __load(self, file, scaling=None):
        """
        Private MATLAB file loading function.

        Returns:
            Dictionary: Dictionary extracted from the MATLAB file.
        """

        import numpy as np
        file = self.base_dir + file
        if file.endswith('.mat'):
            data = loadmat(file)
        elif file.endswith('.npz'):
            data = dict(np.load(file))

        if scaling == None:
            return data
        elif scaling == 'standard':
            self.x_scaler = StandardScaler().fit(data['x'])
            self.y_scaler = StandardScaler().fit(data['y'])
            # vis_scaler = StandardScaler().fit(data['visibility'])

            data['x'] = self.x_scaler.transform(data['x'])
            data['y'] = self.y_scaler.transform(data['y'])
            # data['visibility'] = vis_scaler.transform(data['visibility'])
            # data['y'] = data['y'] / self.y_feature_std[None,:]

            return data
        elif scaling == 'minmax':
            self.x_scaler = MinMaxScaler().fit(data['x'])
            self.y_scaler = MinMaxScaler().fit(data['y'])
            # vis_scaler = MinMaxScaler().fit(data['visibility'])

            data['x'] = self.x_scaler.transform(data['x'])
            data['y'] = self.y_scaler.transform(data['y'])
            # data['visibility'] = vis_scaler.transform(data['visibility'])

            return data

    def __loadAll(self, is_train, scaling=None):
        """
        Private loading function to load data from all MATLAB files.

        Returns:
            Numpy nd-array: Numpy array extracted from all MATLAB files.
        """
        files = sorted(os.listdir(self.base_dir))
        all_data = defaultdict(lambda : np.empty((0,13)))
        print("Fetching Data...")
        for i in tqdm(range(len(files))):
            data = self.__load(files[i])
            if (is_train and data['train'] != 1) or (not is_train and data['train'] == 1):
                continue

            self.video_lengths.append(data['nframes'][0][0])

            x_min, y_min, x_max, y_max = self.getBoundingBox(data['x'], data['y'])
            bBoxLength = max(x_max - x_min, y_max - y_min) + 1e-8

            # bBoxLengths.append(bBoxLength)
            bBoxParams = [x_min, y_min, x_max, y_max, bBoxLength]
            self.bBoxParams.append(bBoxParams)

            data['x'] = self.normalize(data['x'], x_min, x_max, bBoxLength)
            data['y'] = self.normalize(data['y'], y_min, y_max, bBoxLength)

            all_data['x'] = np.concatenate((all_data['x'], data['x']))
            all_data['y'] = np.concatenate((all_data['y'], data['y']))
            all_data['visibility'] = np.concatenate((all_data['visibility'], data['visibility']))

        for i in range(1, len(self.video_lengths)):
            self.video_lengths[i] = int(self.video_lengths[i]) + int(self.video_lengths[i-1])


        if scaling == None:
            return all_data
        elif scaling == 'standard':
            self.x_scaler = StandardScaler().fit(all_data['x'])
            self.y_scaler = StandardScaler().fit(all_data['y'])
            # vis_scaler = StandardScaler().fit(all_data['visibility'])

            all_data['x'] = self.x_scaler.transform(all_data['x'])
            all_data['y'] = self.y_scaler.transform(all_data['y'])
            # all_data['visibility'] = vis_scaler.transform(all_data['visibility'])

        elif scaling == 'minmax':
            self.x_scaler = MinMaxScaler().fit(all_data['x'])
            self.y_scaler = MinMaxScaler().fit(all_data['y'])
            # vis_scaler = MinMaxScaler().fit(all_data['visibility'])

            all_data['x'] = self.x_scaler.transform(all_data['x'])
            all_data['y'] = self.y_scaler.transform(all_data['y'])
            # all_data['visibility'] = vis_scaler.transform(all_data['visibility'])

        self.save_scalers()
        return all_data

    def getBoundingBox(self, x, y):
        return np.min(x), np.min(y), np.max(x), np.max(y)

    def normalize(self, coord, coord_min, coord_max, bounding_box_length):
        coord_mid = coord_min + (coord_max - coord_min) / 2.0
        return (coord - coord_mid) / (bounding_box_length / 2.0)

    def save_scalers(self):
        x_scaler_file = "X_scaler.save"
        Y_scaler_file = "Y_scaler.save"
        joblib.dump(self.x_scaler, x_scaler_file)
        joblib.dump(self.y_scaler, Y_scaler_file)

    def getJointsData(self, withVisibility=True):
        import pandas as pd
        columns = lambda a: [str(a) + str(i+1) for i in range(self.data['x'].shape[1])]
        if withVisibility:
            df = pd.concat([pd.DataFrame(self.data['x'], columns = columns('x')),
                                pd.DataFrame(self.data['y'], columns = columns('y')),
                                pd.DataFrame(self.data['visibility'], columns = ["visibility" + str(i+1) for i in range(self.data['y'].shape[1])])],
                                axis = 1)
        else:
            df = pd.concat([pd.DataFrame(self.data['x'], columns = columns('x')),
                                pd.DataFrame(self.data['y'], columns = columns('y'))],
                                axis = 1)
        del self.data
        return df

    def getRandomTrainingSet(self, seq_len, batch_size):
        """
        Get Random sequence because want to train the model invariant of the starting frame.

        """
        import torch
        import random
        Jointsdata = self.getJointsData()
        X_train = []
        y_train = []
        for batch_index in range(batch_size):
            start_index = random.randint(0, self.data_len - seq_len)
            end_index = start_index + seq_len + 1
            chunk = Jointsdata[start_index:end_index].values
            X_train.append(chunk[:-1])
            y_train.append(chunk[1:])
        X_train = Variable(torch.Tensor(X_train))
        y_train = Variable(torch.Tensor(y_train))
        cuda = True
        if cuda:
            X_train = X_train.cuda()
            y_train = y_train.cuda()
        return X_train, y_train

    def getSequences(self, seq_length, withVisibility=True):
        """
        For a file with K frames, we generate K sequences by varying
        the starting frame. We skip frames when generating sequences since adjacent
        frames contain similar poses. The number of frames skipped is video-dependent:
        Given a sampled starting frame, we always generate a sequence of length 16,
        where we skip every (K − 1) = 15 frames in the raw sequence after the sampled starting frame.

        Note:
            Once we surpass the end frame, we will repeat the last frame collected
            until we obtain 16 frames. This is to force the forecasting to learn to
            “stop” and remain at the ending pose once an action has completed.

        Args:
            seq_length (int): Number of frames to consider in a sequence.
                              Basically, the number of time steps to unroll the LSTM network.

        Returns:
            Numpy nd-array: Sequences with varying starting frame
            Output shape: (Total number of frames x Sequence length x Input dimension)
        """
        import torch
        Jointsdata = self.getJointsData(withVisibility)
        dict = np.zeros((self.data_len, seq_length, Jointsdata.shape[1]))
        print("Creating Sequences...")
        for i in tqdm(range(self.data_len)):
            sequence = []
            j = i
            n_frames = 0
            while n_frames != seq_length:
                if j < self.data_len:
                    sequence.append(Jointsdata[j:j+1].values.flatten())
                    n_frames += 1
                    j = j + ((self.data_len - 1)//15)
                elif j == self.data_len:
                    sequence.append(Jointsdata[j-1:j].values.flatten())
                    n_frames += 1
                elif j > self.data_len:
                    sequence.append(Jointsdata[-1:].values.flatten())
                    n_frames += 1
            dict[i,:,:] = np.array(sequence, dtype = np.float16)
        return dict

    def getStridedSequences(self, seq_length, stride=1, withVisibility=True):
        """
        Create sequences of sequence length = seq_length using a strided sliding
        window without skipping frames.
        Gave better results than when intermediate frames were skipped.

        Args:
            seq_length (int): Number of frames to consider in a sequence.
                              Basically, the number of time steps to unroll the LSTM network.

        Returns:
            Numpy nd-array: Sequences with varying starting frame
            Output shape: (Total number of frames x Sequence length x Input dimension)
        """

        import torch
        Jointsdata = self.getJointsData(withVisibility)
        dict = np.empty((0, seq_length, Jointsdata.shape[1]))
        bBoxForSeq = []
        print("Creating sequences...")
        i=0
        pointer = 0
        while i < self.video_lengths[-1] - seq_length:
            sequence = []
            j = i
            n_frames = 0
            while n_frames != seq_length:
                sequence.append(Jointsdata[j:j+1].values.flatten())
                n_frames += 1
                j += 1
            # print(np.array(sequence).shape)
            dict = np.concatenate((dict, np.array(sequence)[None, :, :]), axis = 0)
            bBoxForSeq.append(self.bBoxParams[pointer])
            if i+seq_length == self.video_lengths[pointer]:
                # i=self.video_lengths[self.video_lengths.index(i+seq_length)]
                i = self.video_lengths[pointer]
                pointer += 1
            else:
                i+=stride
        return dict, bBoxForSeq

if __name__ == '__main__':
    # x = MPIIData(base_dir = '/home/hrishi/1Hrishi/0Thesis/Data/').load(file = 'mpii_human_pose_v1_u12_1.mat')['RELEASE']
    data_stream = PennActionData(base_dir = '/media/hrishi/OS/1Hrishi/1Cheese/0Thesis/Data/Penn_Action/preprocessed/labels/', is_train = False, scaling = None)
    # print(np.var(data_stream.data['x']))
    sequences, bBoxForSequences = data_stream.getStridedSequences(seq_length = 16, withVisibility = False)
    # print(sequences.shape)
    np.save('Normalized_Test_16SL_26F_Sequences.npy', sequences)
    np.save('Normalized_Test_BBOX_Params.npy', bBoxForSequences)
    # data_stream.visualize(sequences)
