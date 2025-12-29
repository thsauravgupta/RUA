import pandas as pd
from pathlib import Path


def main():
    # Project root: RUA
    base_dir = Path(__file__).resolve().parent.parent

    input_path = base_dir / "data" / "raw" / "IMDB Dataset.csv"
    output_path = base_dir / "data" / "processed" / "IMDB_cleaned.csv"

    # Safety check
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    if not {"review", "sentiment"}.issubset(df.columns):
        raise ValueError(
            f"Expected columns {{'review','sentiment'}}, found {set(df.columns)}"
        )

    df = df.drop_duplicates()
    df = df.dropna(subset=["review", "sentiment"])

    df["review"] = (
        df["review"]
        .str.lower()
        .str.replace("<br />", " ", regex=False)
        .str.replace("http", " ", regex=False)
        .str.replace(r"[^a-z\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    df = df[df["review"].str.len() > 0]

    df.to_csv(output_path, index=False)

    print("Cleaned dataset saved to:", output_path)
    print("Final shape:", df.shape)


if __name__ == "__main__":
    main()
