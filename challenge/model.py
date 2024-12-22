import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from datetime import datetime
from typing import Tuple, Union, List

import json
from dotenv import load_dotenv

load_dotenv()

THRESHOLD_IN_MINUTES = int(os.getenv("THRESHOLD_IN_MINUTES"))
MODEL_FILE_NAME = os.getenv("MODEL_FILE_NAME")
DATA_PATH = os.getenv("DATA_PATH")

# The 10 features the Data Scientist (DS) decided to keep
TOP_10_FEATURES = [
    "OPERA_Latin American Wings",
    "MES_7",
    "MES_10",
    "OPERA_Grupo LATAM",
    "MES_12",
    "TIPOVUELO_I",
    "MES_4",
    "MES_11",
    "OPERA_Sky Airline",
    "OPERA_Copa Air",
]

class ModelPersistence:
    """
    Maneja la carga y guardado del modelo en disco.
    Si el archivo no existe, retorna None.
    """
    @staticmethod
    def load_model(filepath: str):
        try:
            with open(filepath, "rb") as fp:
                model = pickle.load(fp)
            return model
        except FileNotFoundError:
            return None

    @staticmethod
    def save_model(model, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as fp:
            pickle.dump(model, fp)

class DelayModel:

    def __init__(self):
        self._features = TOP_10_FEATURES
        self._model = ModelPersistence.load_model(MODEL_FILE_NAME)
        
        if self._model is None:
            self._train_from_scratch(DATA_PATH)

    def _train_from_scratch(self, data_path: str) -> None:
        """
        Read the CSV from data_path, do preprocessing
        and train the LogisticRegression model. Then save the model.
        """
        if not os.path.exists(data_path):
            return

        data = pd.read_csv(data_path)
        features, target = self.preprocess(data, target_column="delay")

        self._model = LogisticRegression(class_weight="balanced", random_state=42)
        if isinstance(target, pd.DataFrame):
            target = target.squeeze()

        self._model.fit(features, target)
        ModelPersistence.save_model(self._model, MODEL_FILE_NAME)

    def get_min_diff(self, row: pd.Series) -> float:
        """
        Calculate the minute difference between 'Fecha-O' and 'Fecha-I'.

        Args:
            row (pd.Series): a single row of the DataFrame.

        Returns:
            float: minute difference between 'Fecha-O' and 'Fecha-I'.
        """
        fecha_o = datetime.strptime(row["Fecha-O"], "%Y-%m-%d %H:%M:%S")
        fecha_i = datetime.strptime(row["Fecha-I"], "%Y-%m-%d %H:%M:%S")
        min_diff = (fecha_o - fecha_i).total_seconds() / 60
        return min_diff

    def preprocess(
        self, data: pd.DataFrame, target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        features = pd.concat(
            [
                pd.get_dummies(data["OPERA"], prefix="OPERA"),
                pd.get_dummies(data["TIPOVUELO"], prefix="TIPOVUELO"),
                pd.get_dummies(data["MES"], prefix="MES"),
            ],
            axis=1,
        )

        for feature in self._features:
            if feature not in features.columns:
                features[feature] = 0

        features = features[self._features]

        if target_column:
            data["min_diff"] = data.apply(self.get_min_diff, axis=1)
            data[target_column] = np.where(
                data["min_diff"] > THRESHOLD_IN_MINUTES, 1, 0
            )
            return features, data[[target_column]]
        else:
            return features

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        if isinstance(target, pd.DataFrame):
            target = target.squeeze()

        self._model = LogisticRegression(class_weight="balanced", random_state=42)
        self._model.fit(features, target)

        ModelPersistence.save_model(self._model, MODEL_FILE_NAME)

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (List[int]): predicted targets.
        """
        if self._model is None:
            self._model = ModelPersistence.load_model(MODEL_FILE_NAME)
            if self._model is None:
                raise ValueError("Model not found.")

        predictions = self._model.predict(features)
        return predictions.tolist()
