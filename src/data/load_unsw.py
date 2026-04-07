from pathlib import Path
import pandas as pd
from src.utils.common import RAW_DIR, INTERIM_DIR, ensure_directories

FILE_NAMES = [
    "Combined.csv",
]


def load_and_combine_unsw() -> pd.DataFrame:
    ensure_directories()
    dataframes = []

    for file_name in FILE_NAMES:
        file_path = RAW_DIR / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"Missing file: {file_path}")

        df = pd.read_csv(file_path)
        dataframes.append(df)

    combined_df = pd.concat(dataframes, axis=0, ignore_index=True)
    output_path = INTERIM_DIR / "5G-NIDD_combined.csv"
    combined_df.to_csv(output_path, index=False)

    print(f"Combined dataset saved to: {output_path}")
    print(f"Combined shape: {combined_df.shape}")
    return combined_df


if __name__ == "__main__":
    load_and_combine_unsw()
    