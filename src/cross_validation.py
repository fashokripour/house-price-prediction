import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

from src.preprocessing_linear import preprocessor

from src.utils import (
    load_dataset,
    prepare_features,
    split_data,
    calculate_metrics
)


df = load_dataset()

X, y = prepare_features(df)

(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test
) = split_data(
    X,
    y
)


# =========================================
# Development data
# Train + Validation
# Test remains untouched
# =========================================

X_dev = pd.concat(
    [X_train, X_val],
    axis=0
).reset_index(drop=True)

y_dev = pd.concat(
    [y_train, y_val],
    axis=0
).reset_index(drop=True)


# =========================================
# Create price groups only for stratification
# =========================================

price_bins = pd.cut(
    y_dev,
    bins=[
        0,
        100000,
        200000,
        400000,
        np.inf
    ],
    labels=[
        "<100k",
        "100k-200k",
        "200k-400k",
        ">400k"
    ],
    include_lowest=True
)


print("\nDevelopment Price Distribution")
print("------------------------------")

print(
    price_bins.value_counts().sort_index()
)


# =========================================
# Stratified 5-Fold
# =========================================

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


fold_results = []


# =========================================
# Cross Validation
# =========================================

for fold, (train_index, val_index) in enumerate(
    skf.split(
        X_dev,
        price_bins
    ),
    start=1
):

    X_fold_train = X_dev.iloc[
        train_index
    ]

    X_fold_val = X_dev.iloc[
        val_index
    ]

    y_fold_train = y_dev.iloc[
        train_index
    ]

    y_fold_val = y_dev.iloc[
        val_index
    ]


    # -------------------------------------
    # Show price distribution in each fold
    # -------------------------------------

    fold_price_bins = pd.cut(
        y_fold_val,
        bins=[
            0,
            100000,
            200000,
            400000,
            np.inf
        ],
        labels=[
            "<100k",
            "100k-200k",
            "200k-400k",
            ">400k"
        ],
        include_lowest=True
    )


    print(f"\nFold {fold}")
    print("========================")

    print("\nValidation price distribution:")
    print(
        fold_price_bins.value_counts().sort_index()
    )


    # -------------------------------------
    # Fresh model for every fold
    # -------------------------------------

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                clone(preprocessor)
            ),
            (
                "model",
                Ridge(
                    alpha=100
                )
            )
        ]
    )


    # -------------------------------------
    # Log target
    # -------------------------------------

    y_fold_train_log = np.log1p(
        y_fold_train
    )


    # -------------------------------------
    # Train
    # -------------------------------------

    model.fit(
        X_fold_train,
        y_fold_train_log
    )


    # -------------------------------------
    # Predict
    # -------------------------------------

    prediction_log = model.predict(
        X_fold_val
    )

    prediction = np.expm1(
        prediction_log
    )

    # Metrics

    metrics = calculate_metrics(
        y_fold_val,
        prediction
    )


    fold_results.append(
        metrics
    )


    print("\nMetrics")
    print("------------------")

    for key, value in metrics.items():

        if key == "R2":

            print(
                f"{key}: {value:.4f}"
            )

        else:

            print(
                f"{key}: {value:.2f}"
            )


# Cross Validation Summary

results_df = pd.DataFrame(
    fold_results
)


print("\n\nCross Validation Summary")
print("========================")


for column in results_df.columns:

    mean_value = results_df[
        column
    ].mean()

    std_value = results_df[
        column
    ].std()


    if column == "R2":

        print(
            f"{column}: "
            f"{mean_value:.4f} "
            f"+/- {std_value:.4f}"
        )

    else:

        print(
            f"{column}: "
            f"{mean_value:.2f} "
            f"+/- {std_value:.2f}"
        )