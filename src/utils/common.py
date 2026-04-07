from pathlib import Path
import random
import numpy as np

RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw" / "unsw_nb15"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"
TABLES_DIR = OUTPUT_DIR / "tables"


def ensure_directories() -> None:
    for path in [
        INTERIM_DIR,
        PROCESSED_DIR,
        FIGURES_DIR,
        MODELS_DIR,
        TABLES_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)