"""
Tests using the new mock data factory.
Shows how to use mock data in pipeline tests.
"""
import pytest
import pandas as pd
import logging

from tests.mock_data_factory import MockDataFactory
from src.ingestion.data_loading import load_data
from src.ingestion.validator import run_all_validations
from src.preprocessing.data_cleaning import clean_data
from src.preprocessing.feature_engineering import engineer_features
from src.models.trainer import ModelTrainer, RandomForestModel

logger = logging.getLogger(__name__)


class TestPipelineWithMocks:
    """Test pipeline using mock data."""
    
    def test_pipeline_with_mock_data(self, tmp_path):
        """Test full pipeline with mock data."""
        # 1. Create mock data
        df = MockDataFactory.create_full_dataset(n_rows=100)
        
        # 2. Save to temporary file
        file_path = tmp_path / "mock_data.csv"
        df.to_csv(file_path, index=False)
        
        # 3. Load
        loaded = load_data(file_path)
        assert len(loaded) == 100
        logger.info(f"✅ Loaded {len(loaded)} rows")
        
        # 4. Validate (add YearEnd)
        loaded['YearEnd'] = loaded['YearStart']
        run_all_validations(loaded)
        logger.info("✅ Validation passed")
        
        # 5. Clean
        cleaned = clean_data(loaded)
        assert len(cleaned) > 50
        logger.info(f"✅ Cleaned: {len(cleaned)} rows")
        
        # 6. Feature engineering
        engineered = engineer_features(cleaned)
        assert 'DataValue' in engineered.columns
        logger.info(f"✅ Feature engineering: {engineered.shape}")
        
        # 7. Model training
        config = {'n_estimators': 10, 'random_state': 42}
        model = RandomForestModel(config=config)
        trainer = ModelTrainer(
            model=model,
            test_size=0.15,
            val_size=0.1,
            do_hyperparameter_tuning=False,
            do_cross_validation=False,
            random_state=42
        )
        trainer.split_data(engineered, target_col='DataValue')
        trainer.train_with_cv(cv_folds=2)
        metrics = trainer.evaluate()
        
        assert 'R2' in metrics
        logger.info(f"✅ Model R2: {metrics['R2']}")
    
    def test_pipeline_with_missing_values(self, tmp_path):
        """Test pipeline handles missing values correctly."""
        df = MockDataFactory.create_with_missing_values(n_rows=100, missing_rate=0.2)
        
        file_path = tmp_path / "mock_data_missing.csv"
        df.to_csv(file_path, index=False)
        
        loaded = load_data(file_path)
        loaded['YearEnd'] = loaded['YearStart']
        cleaned = clean_data(loaded)
        
        # After cleaning, there should be no missing values
        assert cleaned.isna().sum().sum() == 0
        logger.info("✅ All missing values handled")
    
    def test_pipeline_with_outliers(self, tmp_path):
        """Test pipeline handles outliers correctly."""
        df = MockDataFactory.create_with_outliers(n_rows=100, outlier_count=5)
        
        file_path = tmp_path / "mock_data_outliers.csv"
        df.to_csv(file_path, index=False)
        
        loaded = load_data(file_path)
        loaded['YearEnd'] = loaded['YearStart']
        cleaned = clean_data(loaded)
        
        # Outliers should be capped
        original_max = df['DataValue'].max()
        cleaned_max = cleaned['DataValue'].max()
        
        assert cleaned_max < original_max
        logger.info(f"✅ Outliers handled: {original_max:.2f} → {cleaned_max:.2f}")
    
    def test_pipeline_with_duplicates(self, tmp_path):
        """Test pipeline removes duplicates correctly."""
        df = MockDataFactory.create_with_duplicates(n_rows=100, duplicate_rate=0.1)
        
        file_path = tmp_path / "mock_data_duplicates.csv"
        df.to_csv(file_path, index=False)
        
        original_len = len(df)
        loaded = load_data(file_path)
        loaded['YearEnd'] = loaded['YearStart']
        cleaned = clean_data(loaded)
        
        # Duplicates should be removed
        assert len(cleaned) < original_len
        assert cleaned.duplicated().sum() == 0
        logger.info(f"✅ Duplicates removed: {original_len} → {len(cleaned)}")
    
    def test_pipeline_constant_columns(self, tmp_path):
        """Test pipeline handles constant columns correctly."""
        df = MockDataFactory.create_with_constant_columns(n_rows=100)
        
        file_path = tmp_path / "mock_data_constant.csv"
        df.to_csv(file_path, index=False)
        
        loaded = load_data(file_path)
        loaded['YearEnd'] = loaded['YearStart']
        cleaned = clean_data(loaded)
        engineered = engineer_features(cleaned)
        
        # Constant columns should be handled (warning but not error)
        assert 'Constant_Topic' in engineered.columns
        logger.info("✅ Constant columns handled")