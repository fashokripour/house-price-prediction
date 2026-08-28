import pandas as pd
import numpy as np
from pathlib import Path
import arff

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.preprocessing import preprocessor
from src.validate_data import validate_dataset



PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "dataset.arff"


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

X = df.drop(
    columns=[
        "SalePrice",
        "Id"
    ]
)

y = df["SalePrice"]


# Train / Validation / Test split

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


print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)



model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            LinearRegression()
        )
    ]
)


# Train

model.fit(
    X_train,
    y_train
)


# Validation evaluation

val_prediction = model.predict(X_val)


val_mae = mean_absolute_error(
    y_val,
    val_prediction
)

val_rmse = np.sqrt(
    mean_squared_error(
        y_val,
        val_prediction
    )
)

val_r2 = r2_score(
    y_val,
    val_prediction
)


print("\nValidation Results")
print("------------------")
print(f"MAE: {val_mae:.2f}")
print(f"RMSE: {val_rmse:.2f}")
print(f"R2: {val_r2:.4f}")