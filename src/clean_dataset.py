import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw" / "questions.csv"
OUT_PATH = BASE_DIR / "data" / "processed" / "questions_clean.csv"

df = pd.read_csv(RAW_PATH)

# Drop rows with missing questions
df = df.dropna(subset=["question1", "question2"])

# Drop exact duplicates
df = df.drop_duplicates()

df.to_csv(OUT_PATH, index=False)

print("Cleaned dataset shape:", df.shape)
