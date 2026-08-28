import arff
import pandas as pd

with open("data/dataset.arff", "r") as file:
    dataset = arff.load(file)

df = pd.DataFrame(
    dataset["data"],
    columns=[attribute[0] for attribute in dataset["attributes"]]
)

print(df.head())
print(df.shape)
print(df.columns)

missing_percent = df.isnull().mean() * 100

print(missing_percent[missing_percent > 0])