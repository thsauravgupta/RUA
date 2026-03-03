import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Configuration
SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

assert TRAIN_RATIO + VAL_RATIO + TEST_RATIO == 1.0

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "processed" / "questions_clean.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

# Cleaned Dataset
df = pd.read_csv(INPUT_PATH)

print(f"Total samples: {len(df)}")

# 80% Train, 20% Temp (to be split into Val and Test)
train_df, temp_df = train_test_split(
    df,
    test_size=(1 - TRAIN_RATIO),
    random_state=SEED,
    shuffle=True
)

# 50% Val, 50% Test from Temp
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,   # split 20% into 10% / 10%
    random_state=SEED,
    shuffle=True
)

# Save splits
train_df.to_csv(OUTPUT_DIR / "train_clean.csv", index=False)
val_df.to_csv(OUTPUT_DIR / "val_clean.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "test_clean.csv", index=False)


# Summary
print("\nDataset split complete:")
print(f"Train: {len(train_df)} ({len(train_df)/len(df):.2%})")
print(f"Validation: {len(val_df)} ({len(val_df)/len(df):.2%})")
print(f"Test: {len(test_df)} ({len(test_df)/len(df):.2%})")

print("\nLabel distribution check (train):")
print(train_df["is_duplicate"].value_counts(normalize=True))

print("\nLabel distribution check (val):")
print(val_df["is_duplicate"].value_counts(normalize=True))

print("\nLabel distribution check (test):")
print(test_df["is_duplicate"].value_counts(normalize=True))
