import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
SEED = 42
NOISE_LEVELS = [0.1, 0.2, 0.3]

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / "data" / "processed" / "train_clean.csv"
OUTPUT_DIR = BASE_DIR / "data" / "noisy"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load training data
df = pd.read_csv(TRAIN_PATH)
n_samples = len(df)

print(f"Training samples: {n_samples}")

np.random.seed(SEED)

# Inject noise and save datasets
for noise in NOISE_LEVELS:
    noisy_df = df.copy()

    n_noisy = int(noise * n_samples)
    noisy_indices = np.random.choice(
        noisy_df.index,
        size=n_noisy,
        replace=False
    )

    # Flip labels
    noisy_df.loc[noisy_indices, "is_duplicate"] = (
        1 - noisy_df.loc[noisy_indices, "is_duplicate"]
    )

    # mark noisy samples (for analysis only)
    noisy_df["is_noisy"] = 0
    noisy_df.loc[noisy_indices, "is_noisy"] = 1

    # Save
    out_path = OUTPUT_DIR / f"train_noise_{int(noise*100)}.csv"
    noisy_df.to_csv(out_path, index=False)

    print(f"Injected {int(noise*100)}% noise → {n_noisy} samples")
