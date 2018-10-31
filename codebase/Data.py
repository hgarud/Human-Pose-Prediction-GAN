import numpy as np
import pandas as pd
from scipy.io import loadmat
from torch.autograd import Variable

class MPIIData(object):
    def __init__(self, base_dir):
        self.base_dir = base_dir
        assert self.base_dir[-1] == '/'

    def load(self, file):
        file = self.base_dir + file
        x = loadmat(file)
        return x

class PennActionData(object):
    def __init__(self, base_dir, file):
        self.base_dir = base_dir
        assert self.base_dir[-1] == '/'
        self.file = file
        self.data = self.__load(file)
        self.data_len = self.__load(file)['nframes'][0,0]

    # private mat-file loading function
    def __load(self, file):
        file = self.base_dir + file
        return loadmat(file)

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

        """ Get Random sequence because want to train
            the model invariant of the starting frame.
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
            2-D Numpy array: Sequences with varying starting frame
            Output shape: (Total number of frames x Sequence length x Input dimension)
        """
        import torch
        Jointsdata = self.getJointsData(withVisibility)
        dict = np.zeros((self.data_len, seq_length, Jointsdata.shape[1]))
        for i in range(self.data_len):
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


if __name__ == '__main__':
    # x = MPIIData(base_dir = '/home/hrishi/1Hrishi/0Thesis/Data/').load(file = 'mpii_human_pose_v1_u12_1.mat')['RELEASE']
    data_stream = PennActionData(base_dir = '/home/hrishi/1Hrishi/0Thesis/Data/Penn_Action/labels/', file = '0758.mat')
    print(data_stream.data_len)
    sequences = data_stream.getSequences(16, withVisibility=False)
    rand_key = np.random.randint(low = 0, high = 663)
    x_test = sequences[rand_key]
    print(sequences.shape, x_test.shape)
