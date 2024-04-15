import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
class Model(ABC):

    @abstractmethod
    def train(self, X_train, y_train):
        # it's an abstract method, we don't need to provide an implementation here
        pass


class LinearRegressionModel(Model):

    def train(self, X_train, y_train, **kwargs):

        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model Training completed")
            return reg

        except Exception as e:
            logging.error(f"Error while training model: {e}")
            raise e


