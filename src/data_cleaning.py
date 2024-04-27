import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

class DataStrategy(ABC):
    @abstractmethod
    def handle_data(self, data:pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreProcessStrategy(DataStrategy):

    def handle_data(self, data:pd.DataFrame) -> pd.DataFrame:

        try:
            # Preprocessing
            label_encoder = LabelEncoder()
            data["diagnosis"] = label_encoder.fit_transform(data["diagnosis"].values)
            return data
        
        except Exception as e:
            logging.error(f"Error in preprocessing the data: {e}")
            raise e

class DataDivisionStrategy(DataStrategy):
    def handle_data(self, data:pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            X = data.drop(["diagnosis"], axis=1)
            y = data["diagnosis"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            X_train = StandardScaler().fit_transform(X_train)
            X_test = StandardScaler().fit_transform(X_test)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error in dividing data into train and test: {e}")
            raise e

class DataCleaning:

    def __init__(self, data:pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error in handling data: {e}")
            raise e

# if __name__ == "__main__":
#     data = pd.read_csv("../data/data.csv")
#     data_cleaning = DataCleaning(data, DataPreProcessStrategy())
#     data_cleaning.handle_data()
