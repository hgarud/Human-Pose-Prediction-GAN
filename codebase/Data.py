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

    # private mat-file loading function
    def __load(self, file):
        file = self.base_dir + file
        return loadmat(file)

    def getJointsData(self):
        data = self.__load(self.file)
        import pandas as pd
        columns = lambda a: [str(a) + str(i+1) for i in range(data['x'].shape[1])]
        df = pd.concat([pd.DataFrame(data['x'], columns = columns('x')),
                            pd.DataFrame(data['y'], columns = columns('y')),
                            pd.DataFrame(data['visibility'], columns = ["visibility" + str(i+1) for i in range(data['y'].shape[1])])],
                            axis = 1)

        return df

    def get_random_training_set(self, seq_len, batch_size):
        import torch
        import random
        data = self.getJointsData()
        data_len = len(data)
        X_train = []
        y_train = []
        for batch_index in range(batch_size):
            start_index = random.randint(0, data_len - seq_len)
            end_index = start_index + seq_len + 1
            chunk = data[start_index:end_index].values
            X_train.append(chunk[:-1])
            y_train.append(chunk[1:])
        X_train = Variable(torch.LongTensor(X_train))
        y_train = Variable(torch.LongTensor(y_train))
        cuda = True
        if cuda:
            X_train = X_train.cuda()
            y_train = y_train.cuda()
        return X_train, y_train

if __name__ == '__main__':
    # x = MPIIData(base_dir = '/home/hrishi/1Hrishi/0Thesis/Data/').load(file = 'mpii_human_pose_v1_u12_1.mat')['RELEASE']
    x, y = PennActionData(base_dir = '/home/hrishi/1Hrishi/0Thesis/Data/Penn_Action/labels/', file = '0758.mat').get_random_training_set(16, 32)
    print(x.shape)
