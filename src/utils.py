import gc
from pathlib import Path
from src.config import DATA_DIR

def print_section(title):
    print("\n")
    print("=" * 70)
    print(title)
    print("=" * 70)

def cleanup():
    gc.collect()

def check_dataset():
    print_section("DATASET DOWNLOAD")
    expected_files = ["train_transaction.csv", "train_identity.csv"]
    if all((DATA_DIR / f).exists() for f in expected_files):
        print("Dataset already exists.")
        return True

    try:
        import kagglehub
        import shutil
        print("Downloading dataset via kagglehub...")
        path = kagglehub.dataset_download("lnasiri007/ieeecis-fraud-detection")
        print("Downloaded to:", path)

        # Copy CSVs into DATA_DIR
        for f in expected_files:
            for candidate in Path(path).rglob(f):
                shutil.copy(candidate, DATA_DIR / f)
                print(f"Copied {f} -> {DATA_DIR}")
                break

        if all((DATA_DIR / f).exists() for f in expected_files):
            print("Dataset ready.")
            return True

        print("CSVs not found in downloaded path:", path)
        return False

    except Exception as e:
        print("kagglehub download failed:", e)
        print("\nEnsure kagglehub is installed:  pip install kagglehub")
        print("And you are logged in:          kagglehub.login()")
        return False
