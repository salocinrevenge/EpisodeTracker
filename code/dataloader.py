import numpy as np
import pandas as pd


class ReyesDataset():
    """
    A dataset loader for data of UCI-HAR (https://doi.org/10.24432/C54S4K) used in the paper:
    "An Analysis of Time-Frequency Consistency in Human Activity Recognition" by Hecker et al.
    the dataset file is consisted by 9 channels, 3 for x y z of total acceleration, body acceleration
    and body gyroscope; 128 samples for each channel; one label for each sample, that could be 
    walking, walking upstairs, walking downstairs, sitting, standing, and lying. This dataset
    was sampled at 50Hz, so each sample has a duration of 2.56 seconds.
    This dataset class loads a csv file with no header, where each row is a sample. The first
    128 columns are the time_steps of the first channel, the next 128 columns are the time_steps of the
    second channel, and so on. The last column is the label of the sample, totalizing 1153 columns.
    The label is a float number from 0.0 to 5.0, representing the activity, by the order mentioned.
    
    """
    def __init__(self, path: str):
        """
        Builder of the ReyesDataset class.
        
        Parameters
        ----------
        path : str
            The path to the csv file with the desired dataset
        
        """
        dataset = pd.read_csv(path, header=None)
        self.X, self.Y = self.convert(dataset)
        self.len = self.X.shape[0]

    def __getitem__(self, index: int):
        """
        Get a sample from the dataset by its index.

        Parameters
        ----------
        index : int
            The index of the desired sample

        Returns
        -------
        tuple
            A tuple with the sample and its label. The sample is a numpy array with
            shape (9, 128) or (channels, time_steps). The label is a integer from 0 to 5.
        
        """
        return self.X[index], self.Y[index]

    def __len__(self):
        """
        Get the length of the dataset.

        Returns
        -------
        int
            The number of samples in the dataset.
        
        """
        return self.len
    
    def convert(self, dataset: pd.DataFrame, ncanais: int = 9, tamanho: int = 128): # dataset is a pandas dataframe
        """
        Convert the dataset from a pandas dataframe to a numpy array.
        
        Parameters
        ----------
        dataset : pd.DataFrame
            The dataset to be converted
        ncanais : int
            The number of channels in the dataset
        tamanho : int
            The number of time_steps in each channel
        
        Returns
        -------
        tuple
            A tuple with the converted dataset. The first element is a numpy array with
            shape (n_samples, n_channels, n_time_steps) with type float64. The second element is a numpy array
            with shape (n_samples,) with type integer.
        
        """
        dataset = np.asarray(dataset)
        X = np.array(dataset[:, :tamanho*ncanais],dtype=np.float64)
        Y = np.array(dataset[:,tamanho*ncanais],dtype=np.long)
        
        X = X.reshape(X.shape[0], ncanais, -1)
        return X,Y