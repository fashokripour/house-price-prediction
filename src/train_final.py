import pandas as pd
import joblib

from sklearn.ensemble import GradientBoostingRegressor

from src.preprocessing import preprocessor

from src.utils import (
    load_dataset,
    prepare_features,
    split_data,
    calculate_metrics
)

from src.error_analysis import (
    create_error_analysis,
    save_error_analysis,
    evaluate_price_ranges
)

# Load data

df = load_dataset()

X, y = prepare_features(df)

# Original split
# Test must stay untouched until here

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

# Combine Train + Validation

X_dev = pd.concat(
    [X_train, X_val],
    axis=0
)

y_dev = pd.concat(
    [y_train, y_val],
    axis=0
)

print("Development samples:", len(y_dev))
print("Test samples:", len(y_test))

# Final model

model = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)

# Preprocessing + Model

from sklearn.pipeline import Pipeline

final_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)

# Train final model

print("\nTraining final model...")

final_model.fit(
    X_dev,
    y_dev
)

print("Training completed.")

# Final Test prediction

test_prediction = final_model.predict(
    X_test
)

# Final Test metrics

metrics = calculate_metrics(
    y_test,
    test_prediction
)


print("\nFinal Test Results")
print("==================")

for key, value in metrics.items():

    if key == "R2":

        print(
            f"{key}: {value:.4f}"
        )

    else:

        print(
            f"{key}: {value:.2f}"
        )

# Error analysis

error_analysis = create_error_analysis(
    X_test,
    y_test,
    test_prediction
)


save_error_analysis(
    error_analysis,
    "results/final_model"
)

# Price range analysis

price_analysis = evaluate_price_ranges(
    error_analysis
)


print("\nFinal Test Price Range Analysis")
print("===============================")

print(
    price_analysis
)

joblib.dump(
    final_model,
    "models/house_price_model.joblib"
)
print("Model saved.")