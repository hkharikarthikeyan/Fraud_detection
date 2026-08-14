from pathlib import Path
import os

RANDOM_STATE = 42

# Since this file resides in src/config.py, the project root is the parent directory
BASE_DIR = Path(
    os.environ.get("FRAUD_PROJECT_DIR", str(Path(__file__).resolve().parent.parent))
)

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
PLOT_DIR = REPORT_DIR / "plots"

# Ensure target directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
