import pandas as pd

df = pd.read_csv("../data/raw/questions.csv")

print(df.shape)
print(df.columns)

print("---------------------")
print(df.isnull().sum()) #missing values

print("---------------------")

#class distribution
print(df['is_duplicate'].value_counts())
print(df['is_duplicate'].value_counts(normalize=True))

