import pandas as pd

class DataProcessor:
    """
    A class to handle data loading and preprocessing.
    """
    def __init__(self, csv_path, features_all, training_split, validation_split):
        self.csv_path = csv_path
        self.features_all = features_all
        self.training_split = training_split
        self.validation_split = validation_split
        self.dataset = None
        self.data_mean = None
        self.data_std = None

    def load_and_preprocess_data(self):
        """
        Loads data from CSV, selects features, sets index, and standardizes.
        """
        df2 = pd.read_csv(self.csv_path)
        features = df2[self.features_all]
        features.index = df2['Datetime']

        self.dataset = features
        self.data_mean = self.dataset[:].mean(axis=0)
        self.data_std = self.dataset[:].std(axis=0)
        self.dataset = (self.dataset - self.data_mean) / self.data_std
        self.dataset = self.dataset.values
        return self.dataset

    def get_standardization_params(self):
        """
        Returns the mean and standard deviation of the data.
        """
        return self.data_mean, self.data_std

    def get_target_data_for_training(self):
        """
        Returns the target data (second column) for the training split.
        """
        if self.dataset is None:
            raise ValueError("Data not loaded. Call load_and_preprocess_data() first.")
        return pd.DataFrame(self.dataset[0:self.training_split, 1])