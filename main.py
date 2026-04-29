from src.utils.common import ensure_directories, set_seed
from src.data.load_unsw import load_and_combine_unsw
from src.data.clean_data import clean_unsw_data
from src.data.preprocess import preprocess_and_save
from src.data.imbalance import apply_smote
from src.features.feature_selection import select_features_with_rf
from src.models.train_supervised import run_supervised_training


def main():
    ensure_directories()
    set_seed()

    print("Step 1: Loading and combining UNSW-NB15 files...")
    df = load_and_combine_unsw()

    print("Step 2: Cleaning dataset...")
    df_clean = clean_unsw_data(df)

    print("Step 3: Preprocessing and splitting...")
    preprocess_and_save(df_clean)

    print("Step 4: Handling class imbalance...")
    apply_smote()

    print("Step 5: Feature selection...")
    select_features_with_rf()

    print("Step 6: Training supervised models...")
    run_supervised_training()

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
    
