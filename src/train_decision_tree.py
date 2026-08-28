import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import arff

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.preprocessing import preprocessor
from src.validate_data import validate_dataset
from src.error_analysis import (
    create_error_analysis,
    evaluate_group,
    plot_actual_prediction
)


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
            DecisionTreeRegressor(
                max_depth=5,
                random_state=42
            )
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

error_analysis = create_error_analysis(
    X_val,
    y_val,
    val_prediction
)

Path("results").mkdir(
    exist_ok=True
)


error_analysis.to_excel(
    "results/decision_tree/all_predictions.xlsx",
    index=False,
    engine="openpyxl"
)

error_analysis.sort_values(
    by="Absolute_Error",
    ascending=False
).head(20).to_excel(
    "results/decision_tree/worst_predictions.xlsx",
    index=False,
    engine="openpyxl"
)


neighborhood_report = evaluate_group(
    error_analysis,
    "Neighborhood"
)

neighborhood_report.to_excel(
    "results/decision_tree/neighborhood_error_report.xlsx",
    engine="openpyxl"
)


quality_report = evaluate_group(
    error_analysis,
    "OverallQual"
)

quality_report.to_excel(
    "results/decision_tree/quality_error_report.xlsx",
    engine="openpyxl"
)

plot_actual_prediction(
    error_analysis,
    "results/decision_tree/actual_vs_prediction.png"
)


print("Error analysis completed")

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

val_mape = (
    np.abs(
        (y_val - val_prediction)
        /
        y_val
    ).mean()
    * 100
)

print("\nValidation Results")
print("------------------")
print(f"MAE: {val_mae:.2f}")
print(f"RMSE: {val_rmse:.2f}")
print(f"R2: {val_r2:.4f}")
print(f"MAPE: {val_mape:.2f}%")