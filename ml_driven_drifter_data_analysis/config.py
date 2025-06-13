#%%

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT ="/Users/1614576/Desktop/ML-driven drifter data analysis"

DATA_DIR = f'{PROJ_ROOT}/data'
RAW_DATA_DIR = f'{PROJ_ROOT}/data/raw'
INTERIM_DATA_DIR = f'{PROJ_ROOT}/data/interim'
PROCESSED_DATA_DIR = f'{PROJ_ROOT}/data/processed'
EXTERNAL_DATA_DIR = f'{PROJ_ROOT}/data/external'

MODELS_DIR = f'{PROJ_ROOT}/models'
REFS_DIR = f'{PROJ_ROOT}/references'
REPORTS_DIR =f'{PROJ_ROOT}/reports'
FIGURES_DIR = f'{PROJ_ROOT}/reports/figures'

# %%
