from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from src.preprocessing import preprocessor

from src.utils import (
    load_dataset,
    prepare_features,
    split_data,
    calculate_metrics
)

from src.error_analysis import (
    create_error_analysis,
    plot_actual_prediction,
    save_error_analysis
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
            LinearRegression()
        )
    ]
)



model.fit(
    X_train,
    y_train
)



val_prediction = model.predict(
    X_val
)



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
    "results/linear_regression"
)