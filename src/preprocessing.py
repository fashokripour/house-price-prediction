# src/preprocessing.py

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder
)


numerical_features = [
    "LotFrontage",
    "LotArea",
    "YearBuilt",
    "YearRemodAdd",
    "MasVnrArea",

    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",

    "1stFlrSF",
    "2ndFlrSF",
    "LowQualFinSF",
    "GrLivArea",

    "GarageYrBlt",
    "GarageCars",
    "GarageArea",

    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "3SsnPorch",
    "ScreenPorch",

    "PoolArea",
    "MiscVal",

    "MoSold",
    "YrSold"
]


categorical_features = [
    "MSZoning",
    "Street",
    "Alley",
    "LandContour",
    "Utilities",
    "LotConfig",

    "Neighborhood",
    "Condition1",
    "Condition2",

    "BldgType",
    "HouseStyle",

    "RoofStyle",
    "RoofMatl",

    "Exterior1st",
    "Exterior2nd",

    "MasVnrType",

    "Foundation",

    "Heating",
    "CentralAir",
    "Electrical",

    "Functional",

    "GarageType",

    "PavedDrive",

    "Fence",
    "MiscFeature",

    "SaleType",
    "SaleCondition"
]


ordinal_features = [
    "OverallQual",
    "OverallCond",

    "ExterQual",
    "ExterCond",

    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",

    "HeatingQC",

    "KitchenQual",

    "FireplaceQu",

    "GarageFinish",
    "GarageQual",
    "GarageCond",

    "PoolQC",

    "LotShape",
    "LandSlope"
]


ordinal_categories = [
    [1,2,3,4,5,6,7,8,9,10],
    [1,2,3,4,5,6,7,8,9,10],

    ["Po","Fa","TA","Gd","Ex"],
    ["Po","Fa","TA","Gd","Ex"],

    ["Po","Fa","TA","Gd","Ex"],
    ["Po","Fa","TA","Gd","Ex"],
    ["No","Mn","Av","Gd"],
    ["Unf","LwQ","Rec","BLQ","ALQ","GLQ"],
    ["Unf","LwQ","Rec","BLQ","ALQ","GLQ"],

    ["Po","Fa","TA","Gd","Ex"],

    ["Po","Fa","TA","Gd","Ex"],

    ["Po","Fa","TA","Gd","Ex"],

    ["Unf","RFn","Fin"],
    ["Po","Fa","TA","Gd","Ex"],
    ["Po","Fa","TA","Gd","Ex"],

    ["Fa","TA","Gd","Ex"],

    ["IR3","IR2","IR1","Reg"],
    ["Sev","Mod","Gtl"]
]


numerical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


ordinal_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OrdinalEncoder(
                categories=ordinal_categories,
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_transformer,
            numerical_features
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        ),
        (
            "ordinal",
            ordinal_transformer,
            ordinal_features
        )
    ]
)