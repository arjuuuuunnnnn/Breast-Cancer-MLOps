import logging
import pandas as pd
from zenml import step 
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig
import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(
        X_train:pd.DataFrame,
        X_test:pd.DataFrame,
        y_train:pd.Series,
        y_test:pd.Series,
        config:ModelNameConfig
    ) -> RegressorMixin:

    try:
        model = None
        if config.model_name == "LinearRegression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            train_model = model_train(X_train, y_train)
            return trained_model

        else:
            raise ValueError(f"MODEL {config.model_name} not supported")

    except Exception as e:
        logging.error(f"Model in training model: {e}")
        raise e
