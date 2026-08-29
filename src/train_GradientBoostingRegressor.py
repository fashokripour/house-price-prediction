from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd

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
            GradientBoostingRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=4,
                random_state=42
            )
        )
    ]
)



model.fit(
    X_train,
    y_train
)

feature_names = (
    model
    .named_steps["preprocessor"]
    .get_feature_names_out()
)


importances = (
    model
    .named_steps["model"]
    .feature_importances_
)


importance_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": importances
    }
)


importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)


print(
    importance_df.head(20)
)

importance_df.to_csv(
    "results/random_forest/feature_importance.csv",
    index=False
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
    "results/GradientBoostingRegressor"
)

price_analysis = evaluate_price_ranges(
    error_analysis
)


print("\nPrice Range Analysis")
print("--------------------")
print(price_analysis)