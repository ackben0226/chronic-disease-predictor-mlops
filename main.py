"""
main.py
Complete ML pipeline: Load → Validate → Clean → Engineer → Train → Save
"""

import pandas as pd
from pathlib import Path
import logging
import sys
import json

# ============================================================
# IMPORTS
# ============================================================

# Core pipeline imports
from src.ingestion.validator import run_raw_data_validations
from src.preprocessing.data_cleaning import clean_data
from src.preprocessing.feature_engineering import engineer_features
from src.models.trainer import ModelTrainer, RandomForestModel

# Model saving/loading
from scripts.save_model import save_model, load_model, list_models, load_metadata

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
file_path = DATA_DIR / "raw" / "Chronic_Disease_Indicators.csv"

logger.info(f"Looking for file at: {file_path}")

if not file_path.exists():
    logger.error(f"CSV file missing at {file_path}")
    raise FileNotFoundError(f"CSV file not found in: {file_path}")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def ensure_all_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all columns in DataFrame are numeric.
    Drops or converts non-numeric columns.
    """
    df = df.copy()
    non_numeric = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if non_numeric:
        logger.warning(f"Found {len(non_numeric)} non-numeric columns")
        
        for col in non_numeric[:10]:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                logger.info(f"Converted '{col}' to numeric")
            except:
                logger.warning(f"Dropping non-numeric column: '{col}'")
                df = df.drop(columns=[col])
    
    remaining = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if remaining:
        logger.warning(f"Dropping remaining non-numeric columns: {remaining[:5]}...")
        df = df.drop(columns=remaining)
    
    return df


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline():
    """Run the complete ML pipeline."""
    
    rf_model = None
    
    try:
        # ============================================================
        # 1. LOAD RAW DATA
        # ============================================================
        logger.info("=" * 60)
        logger.info("STEP 1: Loading raw data")
        logger.info("=" * 60)
        
        df = pd.read_csv(file_path, low_memory=False)
        logger.info(f"Raw data loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

        # ============================================================
        # 2. VALIDATE
        # ============================================================
        logger.info("=" * 60)
        logger.info("STEP 2: Validating data")
        logger.info("=" * 60)
        
        run_raw_data_validations(df)

        # ============================================================
        # 3. CLEAN
        # ============================================================
        logger.info("=" * 60)
        logger.info("STEP 3: Cleaning data")
        logger.info("=" * 60)
        
        cleaned_df = clean_data(df)
        logger.info(f"Data cleaned. Shape: {cleaned_df.shape}")

        # ============================================================
        # 4. FEATURE ENGINEERING
        # ============================================================
        logger.info("=" * 60)
        logger.info("STEP 4: Feature engineering")
        logger.info("=" * 60)
        
        engineered_df = engineer_features(cleaned_df)
        logger.info(f"Feature engineering complete. Shape: {engineered_df.shape}")

        engineered_df = ensure_all_numeric(engineered_df)
        logger.info(f"After numeric check. Shape: {engineered_df.shape}")

        if 'DataValue' not in engineered_df.columns:
            logger.error("🚨 'DataValue' column is MISSING!")
            raise ValueError("'DataValue' column missing after feature engineering!")

        # ============================================================
        # 5. SAVE PROCESSED DATA
        # ============================================================
        logger.info("=" * 60)
        logger.info("STEP 5: Saving processed data")
        logger.info("=" * 60)
        
        processed_path = DATA_DIR / "processed" / "Chronic_Disease_Indicators_Engineered.csv"
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        engineered_df.to_csv(processed_path, index=False)
        logger.info(f"Saved engineered data to: {processed_path}")

        # ============================================================
        # 6. MODEL TRAINING
        # ============================================================
        logger.info("=" * 60)
        logger.info("STEP 6: Training model")
        logger.info("=" * 60)

        TARGET_COLUMN = "DataValue"

        DROP_COLS = [
            'DataValueAlt', 'LowConfidenceLimit', 'HighConfidenceLimit',
            'DataValueType', 'DataValueUnit', 'DataValueFootnoteSymbol',
            'DatavalueFootnote', 'DataValueTypeID', 'DataValue_Missing_Flag',
            'CI_Missing_Flag', 'DataValue_Extreme_Flag',
        ]
        DROP_COLS = [col for col in DROP_COLS if col in engineered_df.columns]
        if DROP_COLS:
            logger.info(f"Dropping {len(DROP_COLS)} leakage columns")

        model_config = {
            'n_estimators': 100,
            'max_depth': 15,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'random_state': 42,
            'n_jobs': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'oob_score': False,
        }

        logger.info("Initializing Random Forest model...")
        rf_model = RandomForestModel(config=model_config)

        trainer = ModelTrainer(
            model=rf_model,
            test_size=0.2,
            val_size=0.1,
            random_state=42,
            do_hyperparameter_tuning=True,
            do_cross_validation=True
        )

        logger.info("Splitting data into train/val/test sets...")
        trainer.split_data(
            engineered_df, 
            target_col=TARGET_COLUMN,
            drop_cols=DROP_COLS
        )

        logger.info("Training the model with cross-validation...")
        
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [10, 15, 20],
            'min_samples_split': [5, 10, 20],
            'min_samples_leaf': [2, 4, 8],
            'max_features': ['sqrt', 'log2'],
        }
        
        trainer.train_with_cv(cv_folds=3, param_grid=param_grid)

        metrics = trainer.evaluate()

        logger.info("=" * 40)
        logger.info("MODEL PERFORMANCE METRICS:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 40)

        importance_df = trainer.get_feature_importance(top_n=10)
        if not importance_df.empty:
            logger.info("Top 10 features:")
            for _, row in importance_df.iterrows():
                logger.info(f"  {row['Feature']}: {row['Importance']:.4f}")

        logger.info("=" * 50)
        logger.info("MODEL TRAINING COMPLETE!")
        logger.info("=" * 50)

        # ============================================================
        # 7. PREPARE MODEL FOR SAVING (ADD FEATURE NAMES AND METRICS)
        # ============================================================
        logger.info("=" * 60)
        logger.info("STEP 7: Preparing model for saving")
        logger.info("=" * 60)

        if hasattr(trainer, 'X_train') and trainer.X_train is not None:
            feature_names = trainer.X_train.columns.tolist()
            logger.info(f"✅ Feature names extracted from trainer: {len(feature_names)} features")
        else:
            feature_names = [col for col in engineered_df.columns if col != TARGET_COLUMN]
            logger.info(f"⚠️ Feature names from engineered_df: {len(feature_names)} features")

        # --- Attach to model ---
        rf_model.feature_names = feature_names
        logger.info(f"✅ Feature names attached to model")

        rf_model.metrics = trainer.metrics if hasattr(trainer, 'metrics') else metrics
        logger.info(f"✅ Metrics attached to model")

        logger.info(f"   Model prepared with {len(feature_names)} features")
        logger.info(f"   First 5 features: {feature_names[:5]}")
        logger.info(f"   Last 5 features: {feature_names[-5:]}")

        # ============================================================
        # 8. SAVE MODEL WITH METADATA
        # ============================================================
        logger.info("=" * 60)
        logger.info("STEP 8: Saving model with metadata")
        logger.info("=" * 60)

        result = save_model(
            model=rf_model,
            model_name="chronic_disease_predictor",
            version="1.0",
            output_dir="models",
            include_dependencies=True
        )

        logger.info(f"✅ Model saved to: {result['model_path']}")
        logger.info(f"✅ Metadata saved to: {result['metadata_path']}")
        logger.info(f"✅ Requirements saved to: {result['requirements_path']}")

        # ============================================================
        # 9. VERIFY METADATA
        # ============================================================
        logger.info("=" * 60)
        logger.info("STEP 9: Verifying metadata")
        logger.info("=" * 60)

        try:
            with open(result['metadata_path'], 'r') as f:
                saved_metadata = json.load(f)
            
            feature_count = len(saved_metadata.get('features', []))
            performance = saved_metadata.get('performance', {})
            
            logger.info(f"✅ Metadata verified successfully!")
            logger.info(f"   Model: {saved_metadata.get('model_name', 'Unknown')}")
            logger.info(f"   Version: {saved_metadata.get('version', 'Unknown')}")
            logger.info(f"   Type: {saved_metadata.get('model_type', 'Unknown')}")
            logger.info(f"   Features: {feature_count}")
            logger.info(f"   Performance: {performance}")
            
            if feature_count > 0:
                logger.info(f"   First 3 features: {saved_metadata['features'][:3]}")
            else:
                logger.warning("⚠️ WARNING: No features found in metadata!")
                logger.warning("   Check that rf_model.feature_names was set before saving")
            
            # Verify the model can be loaded
            logger.info("=" * 60)
            logger.info("Testing model load...")
            logger.info("=" * 60)
            
            test_model = load_model("chronic_disease_predictor", "1.0", "models")
            logger.info("✅ Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Metadata verification failed: {e}")

        # List all saved models
        models = list_models()
        logger.info(f"All saved models: {len(models)}")
        for m in models:
            logger.info(f"  - {m['name']} v{m['version']} ({m['size_mb']:.2f} MB)")

        logger.info("=" * 60)
        logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        
        return rf_model, trainer, metrics

    except Exception as e:
        logger.exception("❌ Pipeline failed!")
        raise


# ============================================================
# LOAD EXISTING MODEL (Optional)
# ============================================================

def load_existing_model():
    """Load a previously saved model."""
    try:
        logger.info("=" * 60)
        logger.info("Loading existing model...")
        logger.info("=" * 60)
        
        model = load_model("chronic_disease_predictor", "1.0", "models")
        logger.info("✅ Model loaded successfully")
        
        metadata = load_metadata("chronic_disease_predictor", "1.0", "models")
        logger.info(f"Model type: {metadata.get('model_type')}")
        logger.info(f"Features: {len(metadata.get('features', []))}")
        logger.info(f"Performance: {metadata.get('performance')}")
        
        return model
        
    except FileNotFoundError as e:
        logger.warning(f"Model not found: {e}")
        logger.warning("Please run the pipeline first to train and save the model.")
        return None
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None


# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("STARTING CHRONIC DISEASE PREDICTOR PIPELINE")
    logger.info("=" * 60)
    
    try:
        model, trainer, metrics = run_pipeline()
        logger.info("✅ Pipeline execution completed")
        
        logger.info("-" * 40)
        logger.info("Testing model loading...")
        logger.info("-" * 40)
        
        loaded_model = load_existing_model()
        
        if loaded_model is not None:
            logger.info("✅ Model loaded successfully!")
            logger.info("   The model is ready for deployment.")
        else:
            logger.warning("⚠️ Model not found (this is OK if you just trained it)")
            logger.warning("   The model was saved during training.")
            
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("PIPELINE SCRIPT COMPLETE")
    logger.info("=" * 60)