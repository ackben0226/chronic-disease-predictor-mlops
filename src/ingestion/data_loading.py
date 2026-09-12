"""
Data loading module for Chronic Disease Indicators dataset.
"""
import pandas as pd
from pathlib import Path
import logging

from src.ingestion.validator import run_all_validations
from src.preprocessing.data_cleaning import clean_data
from src.preprocessing.feature_engineering import engineer_features

logger = logging.getLogger(__name__)


def load_data(file_path: Path = None) -> pd.DataFrame:
    """
    Load the Chronic Disease Indicators dataset.
    
    Args:
        file_path: Path to the CSV file. If None, uses default path.
    
    Returns:
        DataFrame with loaded data.
    
    Raises:
        FileNotFoundError: If the CSV file doesn't exist.
    """
    if file_path is None:
        BASE_DIR = Path(__file__).resolve().parent.parent.parent
        DATA_DIR = BASE_DIR / "data"
        file_path = DATA_DIR / "raw" / "Chronic_Disease_Indicators.csv"
    
    logger.info(f"Looking for file at: {file_path}")
    
    if not file_path.exists():
        logger.error(f"CSV file missing at {file_path}")
        raise FileNotFoundError(f"CSV file not found in: {file_path}")
    
    # Load raw data
    df = pd.read_csv(file_path, low_memory=False)
    logger.info(f"Raw data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    return df


def run_full_pipeline(file_path: Path = None) -> pd.DataFrame:
    """
    Run the full data pipeline: load, validate, clean, engineer.
    
    Args:
        file_path: Path to the CSV file. If None, uses default path.
    
    Returns:
        Engineered DataFrame ready for modeling.
    
    Raises:
        Exception: If any step in the pipeline fails.
    """
    try:
        # 1. Load
        df = load_data(file_path)
        
        # 2. Validate
        run_all_validations(df)
        
        # 3. Clean
        cleaned_df = clean_data(df)
        logger.info(f"Data cleaned. Shape: {cleaned_df.shape}")
        
        # 4. Feature Engineering
        engineered_df = engineer_features(cleaned_df)
        logger.info(f"Feature engineering complete. Shape: {engineered_df.shape}")
        
        # 5. Save
        if file_path is None:
            BASE_DIR = Path(__file__).resolve().parent.parent.parent
            DATA_DIR = BASE_DIR / "data"
        else:
            DATA_DIR = file_path.parent.parent
        
        processed_path = DATA_DIR / "processed" / "Chronic_Disease_Indicators_Engineered.csv"
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        engineered_df.to_csv(processed_path, index=False)
        logger.info(f"Saved engineered data to: {processed_path}")
        
        return engineered_df
        
    except Exception as e:
        logger.exception("Pipeline failed with an unexpected error!")
        raise


# --- Script execution ---
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s, %(message)s"
    )
    
    run_full_pipeline()