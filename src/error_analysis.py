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

def save_error_analysis(
    data,
    output_dir
):

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    data.to_excel(
        output_dir / "all_predictions.xlsx",
        index=False,
        engine="openpyxl"
    )


    data.sort_values(
        by="Absolute_Error",
        ascending=False
    ).head(20).to_excel(
        output_dir / "worst_predictions.xlsx",
        index=False,
        engine="openpyxl"
    )


    neighborhood_report = evaluate_group(
        data,
        "Neighborhood"
    )


    neighborhood_report.to_excel(
        output_dir / "neighborhood_error_report.xlsx",
        engine="openpyxl"
    )


    quality_report = evaluate_group(
        data,
        "OverallQual"
    )


    quality_report.to_excel(
        output_dir / "quality_error_report.xlsx",
        engine="openpyxl"
    )

    plot_actual_prediction(
        data,
        output_dir / "actual_vs_prediction.png"
    )


def evaluate_price_ranges(data):

    bins = [
        0,
        100000,
        200000,
        400000,
        float("inf")
    ]

    labels = [
        "<100k",
        "100k-200k",
        "200k-400k",
        ">400k"
    ]

    data = data.copy()

    data["Price_Range"] = pd.cut(
        data["Actual"],
        bins=bins,
        labels=labels
    )


    result = (
        data
        .groupby("Price_Range", observed=False)
        .apply(
            lambda x: pd.Series(
                {
                    "Count": len(x),

                    "Average_Price":
                    x["Actual"].mean(),

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

                    "MAPE":
                    x["Percentage_Error"].mean()
                }
            )
        )
    )

    return result