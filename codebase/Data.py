import numpy as np
import pandas as pd
from scipy.io import loadmat

class MPIIData(object):
    def __init__(self, base_dir):
        self.base_dir = base_dir
        assert self.base_dir[-1] == '/'

    def load(self, file):
        file = self.base_dir + file
        x = loadmat(file)
        return x

class PennActionData(object):
    def __init__(self, base_dir):
        self.base_dir = base_dir
        assert self.base_dir[-1] == '/'

    # private mat-file loading function
    def __load(self, file):
        file = self.base_dir + file
        return loadmat(file)

    def getJointsData(self, file):
        data = self.__load(file)
        import pandas as pd
        columns = lambda a: [str(a) + str(i+1) for i in range(data['x'].shape[1])]
        return pd.concat([pd.DataFrame(data['x'], columns = columns('x')),
                            pd.DataFrame(data['y'], columns = columns('y')),
                            pd.DataFrame(data['visibility'], columns = ["visibility" + str(i+1) for i in range(data['y'].shape[1])])],
                            axis = 1)

if __name__ == '__main__':
    # x = MPIIData(base_dir = '/home/hrishi/1Hrishi/0Thesis/Data/').load(file = 'mpii_human_pose_v1_u12_1.mat')['RELEASE']
    x = PennActionData(base_dir = '/home/hrishi/1Hrishi/0Thesis/Data/Penn_Action/labels/').getJointsData(file = '0758.mat')
    print(x)
