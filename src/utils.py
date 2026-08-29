from pathlib import Path
import numpy as np
import pandas as pd
import arff

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.validate_data import validate_dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "dataset.arff"



def load_dataset():

    with open(DATA_PATH, "r") as file:
        dataset = arff.load(file)


    df = pd.DataFrame(
        dataset["data"],
        columns=[
            attribute[0]
            for attribute in dataset["attributes"]
        ]
    )


    validate_dataset(df)

    return df



def prepare_features(df):

    X = df.drop(
        columns=[
            "SalePrice",
            "Id"
        ]
    )

    y = df["SalePrice"]

    return X, y



def split_data(
    X,
    y
):

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )


    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42
    )


    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )



def calculate_metrics(
    y_true,
    y_pred
):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )


    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )


    r2 = r2_score(
        y_true,
        y_pred
    )


    mape = (
        np.abs(
            (y_true - y_pred)
            /
            y_true
        ).mean()
        *
        100
    )


    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape
    }