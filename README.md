# House Price Prediction

## Project Overview

This project develops and evaluates machine-learning regression models for predicting residential house sale prices using the **Ames Housing dataset**.

The dataset was obtained from **OpenML (Dataset ID: 42165)**:

https://www.openml.org/d/42165

The dataset contains residential property attributes represented by numerical, nominal categorical, and ordinal features. `SalePrice` is used as the prediction target.

The goal of this project was not simply to obtain the best score on one validation split. Instead, the project compares several regression approaches, analyzes their errors, evaluates model stability with cross-validation, and selects the final model based on both predictive performance and robustness.

## Data Preparation

The prediction target is `SalePrice`, and the `Id` column is removed from the model features.

The original dataset is split into:

- **70% training data**
- **15% validation data**
- **15% test data**

The training set is used to fit candidate models. The validation set is used to compare models and make hyperparameter decisions. The test set is kept untouched during model development and is used only after the final model has been selected.

After model selection, the training and validation sets are combined. The selected model is then retrained on this complete development set and evaluated once on the held-out test set.

## Preprocessing

Preprocessing is implemented with scikit-learn pipelines so that the same transformations used during training can also be applied consistently during prediction.

### General Preprocessing

Features are separated into numerical, nominal categorical, and ordinal groups.

- Missing numerical values are imputed using the median.
- Missing categorical and ordinal values are imputed using the most frequent value.
- Nominal categorical features are transformed with one-hot encoding.
- Ordered categorical features are transformed with ordinal encoding using explicitly defined category orders.
- Unknown categories are handled so that unseen values do not break the prediction pipeline.

This preprocessing pipeline is used with the tree-based models.

### Ridge-Specific Preprocessing

Ridge Regression uses a separate preprocessing pipeline. In addition to imputation and encoding, numerical features are standardized using `StandardScaler`.

Scaling is important for Ridge because L2 regularization penalizes coefficient magnitudes. If features are measured on very different scales, the regularization penalty can affect them unevenly.

The Ridge experiments also tested a logarithmic transformation of the target:

```python
y_log = np.log1p(y)
```

Predictions are converted back to the original price scale with:

```python
prediction = np.expm1(prediction_log)
```

This transformation produced a strong result on the original validation split, although cross-validation later exposed an important stability problem.

## Evaluation Metrics

Four metrics are used to evaluate the regression models:

- **MAE (Mean Absolute Error):** the average absolute difference between predicted and actual prices.
- **RMSE (Root Mean Squared Error):** similar to MAE, but gives a stronger penalty to large errors.
- **R²:** measures how much of the variation in sale prices is explained by the model.
- **MAPE (Mean Absolute Percentage Error):** measures prediction error relative to the actual sale price.

Error analysis is also performed across different price ranges to investigate whether a model behaves differently for lower-, middle-, and higher-priced houses.

## 1. Baseline: Linear Regression

Linear Regression was selected as the baseline because it provides a simple reference point for determining whether more complex models actually improve prediction quality.

### Baseline Validation Result

| MAE | RMSE | R² | MAPE |
|---:|---:|---:|---:|
| 19,133.03 | 29,286.41 | 0.8634 | Not recorded |

The baseline produced a reasonable result, but error analysis showed that its predictions were not equally reliable across all observations. This motivated experiments with models capable of representing nonlinear relationships and feature interactions.

## 2. Decision Tree Regressor

A Decision Tree recursively splits the feature space into smaller regions and predicts a value for observations reaching each final leaf.

**Strengths:** Decision Trees can learn nonlinear relationships and feature interactions and have a relatively intuitive structure.

**Weaknesses:** A single tree can overfit easily and can be sensitive to changes in the training sample.

Across the five recorded Decision Tree validation experiments:

| Experiments | Mean MAE | Mean RMSE | Mean R² | Mean MAPE |
|---:|---:|---:|---:|---:|
| 5 | 27,410.25 | 41,604.52 | 0.7218 | 16.70% |

The Decision Tree performed substantially worse than the baseline and showed the limitations of relying on a single tree. This motivated the use of Random Forest.

## 3. Random Forest Regressor

Random Forest trains multiple Decision Trees using randomized samples and feature subsets and then averages their predictions.

The idea was to preserve the ability of trees to learn nonlinear relationships while reducing the instability of an individual tree.

**Strengths:** Random Forest can model nonlinear relationships and interactions, reduces the variance of a single Decision Tree, and is generally more robust than an individual tree.

**Weaknesses:** It is less interpretable than a single tree, requires more computation, and increasing the number of trees does not necessarily continue improving generalization.

Across five recorded Random Forest validation experiments:

| Experiments | Mean MAE | Mean RMSE | Mean R² | Mean MAPE |
|---:|---:|---:|---:|---:|
| 5 | 16,479.80 | 24,777.35 | 0.9024 | 10.48% |

The best recorded Random Forest configuration used:

```python
RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
```

Its validation result was:

| MAE | RMSE | R² | MAPE |
|---:|---:|---:|---:|
| 15,982.33 | 24,179.70 | 0.9069 | 10.11% |

The substantial improvement over a single Decision Tree suggested that tree ensembles were well suited to the dataset. This led to the next experiment: Gradient Boosting.

## 4. Gradient Boosting Regressor

Unlike Random Forest, where trees are trained largely independently and their predictions are averaged, Gradient Boosting builds trees sequentially. Each new stage attempts to reduce errors left by the previous stages.

**Strengths:** Gradient Boosting performs strongly on many structured/tabular datasets, captures nonlinear relationships and feature interactions, and can progressively correct residual errors.

**Weaknesses:** It is sensitive to hyperparameter choices, sequential training is less parallelizable than Random Forest, and excessive boosting can reduce generalization performance.

Across eight recorded Gradient Boosting validation experiments:

| Experiments | Mean MAE | Mean RMSE | Mean R² | Mean MAPE |
|---:|---:|---:|---:|---:|
| 8 | 16,154.32 | 24,281.46 | 0.9008 | 10.17% |

The strongest validation configuration was:

```python
GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)
```

Its validation result was:

| MAE | RMSE | R² | MAPE |
|---:|---:|---:|---:|
| 15,660.00 | 23,123.78 | 0.9149 | 9.78% |

An experiment with 700 estimators produced substantially worse validation performance. This was useful evidence that simply increasing model complexity was not necessarily beneficial.

## 5. Ridge Regression

After evaluating the tree-based models, Ridge Regression was explored as a regularized linear alternative. This allowed the project to test whether regularization, feature scaling, and transformation of the target could improve on the original linear baseline.

Ridge extends Linear Regression by adding an L2 penalty to the objective function. This discourages excessively large coefficients and can improve behavior when transformed features are numerous or correlated.

A dedicated preprocessing pipeline with feature scaling was used for these experiments. A logarithmic transformation of `SalePrice` was also tested.

Across the seven Ridge experiments for which complete metrics were preserved:

| Experiments | Mean MAE | Mean RMSE | Mean R² | Mean MAPE |
|---:|---:|---:|---:|---:|
| 7 | 17,638.10 | 25,462.07 | 0.8930 | 10.97% |

The strongest single validation result used:

- `alpha=100`
- scaled preprocessing
- a log-transformed target

Its result was:

| MAE | RMSE | R² | MAPE |
|---:|---:|---:|---:|
| **15,457.49** | **21,952.28** | **0.9233** | **9.75%** |

This was better than the best Gradient Boosting result on the original validation split. Based only on that split, Ridge appeared to be the strongest candidate.

However, selecting a final model from a single validation split can hide instability. Cross-validation was therefore introduced before making the final model-selection decision.

## 6. Cross-Validation and Model Stability

A 5-fold cross-validation procedure was used to evaluate whether model performance remained reliable when the development data was divided differently.

Because `SalePrice` is continuous, the target was temporarily divided into four price ranges for stratification:

- below 100k
- 100k–200k
- 200k–400k
- above 400k

These bins were used only to construct folds with broadly similar target-price distributions. The regression models continued to predict the continuous `SalePrice` target.

### Ridge Stability Analysis

Ridge with `alpha=100` and a log-transformed target had produced the strongest result on the original validation split. Stratified cross-validation, however, exposed substantial sensitivity to unusual observations.

One fold produced:

| MAE | RMSE | R² | MAPE |
|---:|---:|---:|---:|
| 21,735.36 | 121,035.74 | **-1.1570** | 13.22% |

Diagnostic analysis found an extreme prediction of approximately **$2.04 million for a house that actually sold for $160,000**.

The observation combined unusually large living area and high overall quality with a much lower actual sale price. The model produced an extreme extrapolated prediction for this unusual observation. Because predictions were made in log-price space and transformed back to the original scale, sufficiently large errors in log space could become very large dollar errors after the inverse transformation.

This showed that Ridge's excellent single-split validation result did not represent sufficiently stable behavior across different data splits.

### Gradient Boosting: Stratified 5-Fold Cross-Validation

The previously selected Gradient Boosting configuration was then evaluated using the same stratified fold design.

| Fold | MAE | RMSE | R² | MAPE |
|---:|---:|---:|---:|---:|
| 1 | 15,891.50 | 33,109.51 | 0.8386 | 8.65% |
| 2 | 15,067.81 | 23,636.58 | 0.8889 | 9.20% |
| 3 | 16,809.16 | 28,804.67 | 0.8681 | 8.80% |
| 4 | 16,076.28 | 28,874.26 | 0.8652 | 9.41% |
| 5 | 14,886.68 | 22,913.80 | 0.9127 | 9.02% |

Cross-validation summary:

| Metric | Mean | Standard Deviation |
|---|---:|---:|
| MAE | 15,746.29 | 784.05 |
| RMSE | 27,467.76 | 4,213.41 |
| R² | 0.8747 | 0.0278 |
| MAPE | 9.02% | 0.30% |

Gradient Boosting remained considerably more consistent across the five folds.

This changed the final model-selection decision. Instead of selecting Ridge because it had the best score on one validation split, Gradient Boosting was selected because it provided a stronger combination of predictive accuracy and stability.

## Final Model

The selected model is:

```python
GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)
```

After model selection was complete, the original training and validation sets were combined. The final pipeline was trained on this complete development set and evaluated once on the untouched test set.

### Final Test Results

| MAE | RMSE | R² | MAPE |
|---:|---:|---:|---:|
| **14,856.23** | **24,121.65** | **0.9235** | **8.44%** |

### Final Test Results by Price Range

| Price Range | Count | Average Price | MAE | RMSE | MAPE |
|---|---:|---:|---:|---:|---:|
| <100k | 20 | 82,859.65 | 10,299.03 | 14,695.77 | 14.03% |
| 100k–200k | 148 | 146,890.76 | 10,743.41 | 14,515.77 | 7.53% |
| 200k–400k | 45 | 257,381.93 | 22,171.75 | 29,199.54 | 8.33% |
| >400k | 6 | 532,583.17 | 76,630.14 | 94,476.07 | 13.07% |

The smallest price ranges contain relatively few test observations, so their subgroup metrics should be interpreted cautiously.

## Key Findings

1. A better score on one validation split does not necessarily identify the best final model.
2. A single Decision Tree performed substantially worse than the ensemble tree models.
3. Random Forest produced a major improvement over an individual Decision Tree.
4. Gradient Boosting provided a strong combination of predictive accuracy and stability.
5. Ridge with a log-transformed target achieved the best single validation result but showed severe sensitivity to unusual observations during cross-validation.
6. Cross-validation materially changed the final model-selection decision.
7. Price-range error analysis showed why aggregate metrics alone are not sufficient for understanding regression performance.
8. The final held-out test result supported the decision to select Gradient Boosting.

## Project Structure

```text
house-price-prediction/
├── data/                       # Dataset excluded from Git
├── models/
│   └── house_price_model.joblib
├── results/
│   ├── linear_regression/
│   ├── decision_tree/
│   ├── random_forest/
│   ├── gradient_boosting/
│   ├── ridge/
│   └── final_model/
├── src/
│   ├── preprocessing.py
│   ├── preprocessing_linear.py
│   ├── utils.py
│   ├── error_analysis.py
│   ├── train_baseline.py
│   ├── train_decision_tree.py
│   ├── train_random_forest.py
│   ├── train_gradient_boosting.py
│   ├── train_ridge.py
│   ├── cross_validation.py
│   └── train_final.py
├── requirements.txt
└── README.md
```

## Libraries Used

The main Python libraries used in the project are:

- NumPy
- pandas
- scikit-learn
- Matplotlib
- liac-arff
- openpyxl
- joblib

Exact package versions are listed in `requirements.txt`.

## Running the Project

Install the dependencies:

```bash
pip install -r requirements.txt
```

Download the Ames Housing dataset from OpenML and place the ARFF file at the path expected by the project's data-loading code.

Run the final training and evaluation script with:

```bash
python -m src.train_final
```

The trained pipeline is saved as:

```text
models/house_price_model.joblib
```

Because preprocessing and Gradient Boosting are stored together in a scikit-learn `Pipeline`, the same preprocessing transformations are applied automatically when the saved model receives new observations with the expected input features.

## AI Assistance Disclosure

AI tools (ChatGPT) were used in this project as a development and learning assistant. They helped with portions of code generation and refactoring, debugging, explanations of machine-learning concepts, and the English writing and organization of this README.

The experiments were executed during the development of the project, and model performance was evaluated using the reported results. Model selection was based on the experimental workflow described above, including validation, error analysis, and cross-validation.

AI assistance is disclosed here to make its role in both the implementation and documentation transparent.

## Future Improvements

Possible future extensions include:

- Build an interactive Streamlit demo for predictions.
- Create a Persian real-estate dataset from publicly available housing listings on Iranian real-estate platforms, subject to the platforms' terms of use and applicable data-collection restrictions. This would include collecting, cleaning, standardizing, and engineering features from local housing data and then retraining and evaluating the prediction pipeline on a dataset that better represents the Iranian housing market.
- Investigate model behavior on unusual and extreme observations.
- Add model explainability to better understand individual predictions.
