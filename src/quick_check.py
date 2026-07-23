import pandas as pd

df = pd.read_excel("data/data.xlsx")

print("Форма таблицы:", df.shape)
print("\n--- df.info() ---")
df.info()

print("\n--- df.head() ---")
print(df.head())

print("\n--- df.dtypes ---")
print(df.dtypes)