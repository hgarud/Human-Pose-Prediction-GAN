import cv2
import numpy as np
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.externals import joblib

class PennActionImageData(object):
    def __init__(self, base_dir, file=None, scaling=None):
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

        if file == None:
            self.data = self.__loadAll(scaling)
            self.data_len = len(self.data['x'])
        else:
            self.file = file
            self.data = self.__load(file, scaling)
            # self.data_len = self.data['nframes'][0,0]

    def __load(self, file, scaling=None):
        """
        Private MATLAB file loading function.

        Returns:
            Dictionary: Dictionary extracted from the MATLAB file.
        """

        frame = str(self.base_dir + file)
        data = cv2.imread(frame)
        # self.video_lengths.append(data['nframes'][0][0])

        if scaling == None:
            return data
        elif scaling == 'standard':
            self.x_scaler = StandardScaler().fit(data)
            self.y_scaler = StandardScaler().fit(data)
            # vis_scaler = StandardScaler().fit(data['visibility'])

            data = self.x_scaler.transform(data)
            data = self.y_scaler.transform(data)
            # data['visibility'] = vis_scaler.transform(data['visibility'])
            # data['y'] = data['y'] / self.y_feature_std[None,:]

            return data
        elif scaling == 'minmax':
            self.x_scaler = MinMaxScaler().fit(data)
            self.y_scaler = MinMaxScaler().fit(data)
            # vis_scaler = MinMaxScaler().fit(data['visibility'])

            data = self.x_scaler.transform(data)
            data = self.y_scaler.transform(data)
            # data['visibility'] = vis_scaler.transform(data['visibility'])

            return data

    def __loadAll(self, scaling=None):
        """
        Private loading function to load data from all MATLAB files.

        Returns:
            Numpy nd-array: Numpy array extracted from all MATLAB files.
        """
        video_dirs = sorted(os.listdir(self.base_dir))
        self.scale_factor = np.zeros((len(video_dirs), 2))
        all_data = np.empty((0,(256*256*3)))
        pointer = 0
        print("Fetching Data...")
        for i in tqdm(range(len(video_dirs))):
            video_frames = sorted(os.listdir(self.base_dir + video_dirs[i]))
            self.video_lengths.append(len(video_frames))
            for frame in video_frames:
                image = self.__load(video_dirs[i] + "/" + frame)
                scaled_image = cv2.resize(image, (256, 256))
                scaled_image = scaled_image.flatten()
                all_data = np.concatenate((all_data, scaled_image[None, :]))
            self.scale_factor[pointer,:] = np.array([image.shape[1], image.shape[0]])
            pointer += 1

        for i in range(1, len(self.video_lengths)):
            self.video_lengths[i] = int(self.video_lengths[i]) + int(self.video_lengths[i-1])


        if scaling == None:
            return all_data
        elif scaling == 'standard':
            self.image_scaler = StandardScaler().fit(all_data)
            # self.y_scaler = StandardScaler().fit(all_data['y'])
            # vis_scaler = StandardScaler().fit(all_data['visibility'])

            all_data = self.image_scaler.transform(all_data)
            # all_data['y'] = self.y_scaler.transform(all_data['y'])
            # all_data['visibility'] = vis_scaler.transform(all_data['visibility'])

        elif scaling == 'minmax':
            self.image_scaler = MinMaxScaler().fit(all_data)
            # self.y_scaler = MinMaxScaler().fit(all_data['y'])
            # vis_scaler = MinMaxScaler().fit(all_data['visibility'])

            all_data = self.image_scaler.transform(all_data)
            # all_data['y'] = self.y_scaler.transform(all_data['y'])
            # all_data['visibility'] = vis_scaler.transform(all_data['visibility'])

        self.save_scalers()
        return all_data

    def save_scalers(self):
        image_scaler_file = "Images_scaler.save"
        # Y_scaler_file = "Y_scaler.save"
        joblib.dump(self.image_scaler, image_scaler_file)
        # joblib.dump(self.y_scaler, Y_scaler_file)

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
        # Jointsdata = self.getJointsData(withVisibility)
        dict = np.empty((0, seq_length, self.data.shape[1]))
        print("Creating sequences...")
        # for i in tqdm(range(0, self.video_lengths[-1] - seq_length, stride)):
        i=0
        pointer = 0
        while i < self.video_lengths[-1] - seq_length:
            sequence = []
            j = i
            n_frames = 0
            while n_frames != seq_length:
                sequence.append(self.data[j:j+1])
                n_frames += 1
                j += 1
            # print(np.array(sequence).shape)
            dict = np.concatenate((dict, np.array(sequence)[None, :, :]), axis = 0)
            if i+seq_length == self.video_lengths[pointer]:
                # i=self.video_lengths[self.video_lengths.index(i+seq_length)]
                print(i)
                i = self.video_lengths[pointer]
                print(i)
                pointer += 1
            else:
                i+=stride
        return dict

if __name__ == '__main__':
    # x = MPIIData(base_dir = '/home/hrishi/1Hrishi/0Thesis/Data/').load(file = 'mpii_human_pose_v1_u12_1.mat')['RELEASE']
    data_stream = PennActionImageData(base_dir = '/media/hrishi/OS/1Hrishi/1Cheese/0Thesis/Data/Penn_Action/preprocessed/frames/', scaling = 'standard')
    # print(data_stream.data.size)
