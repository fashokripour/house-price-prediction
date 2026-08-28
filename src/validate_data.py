import pandas as pd


def validate_dataset(df, expected_features=None):

    print("Dataset shape:")
    print(df.shape)


    print("\nDuplicate rows:")
    print(df.duplicated().sum())


    print("\nDuplicate columns:")
    print(df.columns[df.columns.duplicated()].tolist())


    print("\nMissing values:")
    missing = df.isnull().sum()
    print(missing[missing > 0])


    print("\nData types:")
    print(df.dtypes)


    if expected_features:
        print("\nMissing expected columns:")
        print(
            set(expected_features) - set(df.columns)
        )

        print("\nUnexpected columns:")
        print(
            set(df.columns) - set(expected_features)
        )


    print("\nUnique values:")
    print(df.nunique())


    print("\nValidation finished.")