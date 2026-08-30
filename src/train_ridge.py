from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
import numpy as np

from src.preprocessing_linear import preprocessor

from src.utils import (
    load_dataset,
    prepare_features,
    split_data,
    calculate_metrics
)

from src.error_analysis import (
    create_error_analysis,
    plot_actual_prediction,
    save_error_analysis,
    evaluate_price_ranges
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

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            Ridge(
                alpha=100
            )
        )
    ]
)

y_train_log = np.log1p(y_train)

model.fit(
    X_train,
    y_train_log
)

val_prediction_log = model.predict(X_val)

val_prediction = np.expm1(
    val_prediction_log
)

# Validation metrics

metrics = calculate_metrics(
    y_val,
    val_prediction
)

print("\nValidation Results")
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


error_analysis = create_error_analysis(
    X_val,
    y_val,
    val_prediction
)

save_error_analysis(
    error_analysis,
    "results/ridge"
)

price_analysis = evaluate_price_ranges(
    error_analysis
)


print("\nPrice Range Analysis")
print("--------------------")

print(
    price_analysis
)