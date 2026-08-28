import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def create_error_analysis(
    X_val,
    y_val,
    predictions
):

    results = X_val.copy()

    results["Actual"] = y_val.values
    results["Prediction"] = predictions

    results["Error"] = (
        results["Actual"]
        -
        results["Prediction"]
    )

    results["Absolute_Error"] = (
        results["Error"].abs()
    )

    results["Percentage_Error"] = (
        results["Absolute_Error"]
        /
        results["Actual"]
        *
        100
    )

    return results



def evaluate_group(
    data,
    group_column
):

    result = (
        data
        .groupby(group_column)
        .apply(
            lambda x: pd.Series(
                {
                    "Count": len(x),

                    "MAE":
                    mean_absolute_error(
                        x["Actual"],
                        x["Prediction"]
                    ),

                    "RMSE":
                    mean_squared_error(
                        x["Actual"],
                        x["Prediction"]
                    ) ** 0.5,

                    "R2":
                    r2_score(
                        x["Actual"],
                        x["Prediction"]
                    ),

                    "Mean_Percentage_Error":
                    x["Percentage_Error"].mean()
                }
            )
        )
    )

    return result.sort_values(
        "Mean_Percentage_Error",
        ascending=False
    )



def plot_actual_prediction(data, output_path):

    plt.figure(
        figsize=(8,6)
    )

    plt.scatter(
        data["Actual"],
        data["Prediction"]
    )

    min_value = min(
        data["Actual"].min(),
        data["Prediction"].min()
    )

    max_value = max(
        data["Actual"].max(),
        data["Prediction"].max()
    )

    plt.plot(
        [min_value, max_value],
        [min_value, max_value]
    )

    plt.xlabel(
        "Actual Price"
    )

    plt.ylabel(
        "Predicted Price"
    )

    plt.title(
        "Actual vs Prediction"
    )

    plt.savefig(
        output_path,
        bbox_inches="tight"
    )

    plt.close()




   