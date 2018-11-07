import numpy as np
import pandas as pd
from scipy.io import loadmat
from torch.autograd import Variable
from tqdm import tqdm

class MPIIData(object):
    def __init__(self, base_dir):
        self.base_dir = base_dir
        assert self.base_dir[-1] == '/'

    def load(self, file):
        file = self.base_dir + file
        x = loadmat(file)
        return x

class PennActionData(object):
    def __init__(self, base_dir, file, scaling=None):
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
        self.file = file
        self.data = self.__load(file, scaling)
        self.data_len = self.data['nframes'][0,0]


    def __load(self, file, scaling=None):
        """
        Private MATLAB file loading function.

        Returns:
            Numpy nd-array: Numpy array extracted from the MATLAB file.
        """

        import numpy as np
        file = self.base_dir + file
        data = loadmat(file)
        if scaling == None:
            return data
        elif scaling == 'standard':
            data['x'] = data['x'] - np.mean(data['x'], axis = 0)[None,:]
            data['x'] = data['x'] / np.std(data['x'], axis = 0)[None,:]
            data['y'] = data['y'] - np.mean(data['y'], axis = 0)[None,:]
            data['y'] = data['y'] / np.std(data['y'], axis = 0)[None,:]
            return data
        elif scaling == 'minmax':
            max = np.amax(data['x'], axis = 0)
            min = np.amin(data['x'], axis = 0)
            data['x'] = (data['x'] - min[None,:]) / (max[None,:] - min[None,:])
            max = np.amax(data['y'], axis = 0)
            min = np.amin(data['y'], axis = 0)
            data['y'] = (data['y'] - min[None,:]) / (max[None,:] - min[None,:])
            return data

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
        the starting frame. We skip frames when generating sequencessince adjacent
        frames containsimilar poses. The number of frames skipped is video-dependent:
        Given a sampled starting frame, we always generate a sequence of length 16,
        where we skip every (K − 1) = 15 frames in the raw sequence after the sampled starting frame.

        Note:
            Once we surpass the end frame, we will repeat the last frame collected
            until we obtain 16 frames. This is to force the forecasting to learn to
            “stop” and remain at the ending pose once an action has completed.

        Args:
            seq_length (int): Number of time steps to unroll the LSTM network.

        Returns:
            Numpy nd-array: Sequences with varying starting frame
            Output shape: (Total number of frames x Sequence length x Input dimension)
        """
        import torch
        Jointsdata = self.getJointsData(withVisibility)
        dict = np.zeros((self.data_len, seq_length, Jointsdata.shape[1]))
        print("Fetching Data...")
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
        import torch
        Jointsdata = self.getJointsData(withVisibility)
        dict = np.zeros((self.data_len - seq_length, seq_length, Jointsdata.shape[1]))
        print("Fetching Data...")
        for i in tqdm(range(0, self.data_len - seq_length, stride)):
            sequence = []
            j = i
            n_frames = 0
            while n_frames != seq_length:
                sequence.append(Jointsdata[j:j+1].values.flatten())
                n_frames += 1
            dict[i,:,:] = np.array(sequence, dtype = np.float16)
        return dict

if __name__ == '__main__':
    # x = MPIIData(base_dir = '/home/hrishi/1Hrishi/0Thesis/Data/').load(file = 'mpii_human_pose_v1_u12_1.mat')['RELEASE']
    data_stream = PennActionData(base_dir = '/home/hrishi/1Hrishi/0Thesis/Data/Penn_Action/labels/', file = '0758.mat', scaling = 'standard')
    print(data_stream.data_len)
    print(np.std(data_stream.data['x'][:,0]))

    continuousFrames = data_stream.getContinuousSequences(5, withVisibility=False)
    print(continuousFrames.shape)
    print(continuousFrames[-6:])
