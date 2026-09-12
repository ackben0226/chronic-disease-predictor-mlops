"""
End-to-end pipeline tests.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TestPipeline:
    """Test the full end-to-end pipeline."""

    def test_end_to_end_pipeline(self, tmp_path):
        """Test that the entire pipeline runs end-to-end."""
        # Create larger test dataset (100 rows to survive cleaning)
        n_rows = 100
        
        # Create varied data to avoid duplicates
        states = ['CA', 'NY', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI']
        topics = ['Diabetes', 'Cancer', 'Heart Disease', 'Stroke', 'Obesity']
        questions = ['Prevalence', 'Rate', 'Mortality', 'Incidence', 'Hospitalization']
        data_value_types = ['Prevalence', 'Rate', 'Prevalence', 'Rate', 'Number']
        stratifications = ['Overall', 'Male', 'Female', 'Age 65+', 'Age 18-64']
        strat_categories = ['Overall', 'Gender', 'Gender', 'Age', 'Age']
        
        # Pre-compute all values to ensure same length
        year_starts = [2020 + (i % 3) for i in range(n_rows)]
        location_abbrs = [states[i % len(states)] for i in range(n_rows)]
        location_descs = [f'State_{i}' for i in range(n_rows)]
        topics_list = [topics[i % len(topics)] for i in range(n_rows)]
        questions_list = [questions[i % len(questions)] for i in range(n_rows)]
        data_values = [str(10 + (i % 50)) for i in range(n_rows)]
        low_confidence = [5.0 + (i % 20) for i in range(n_rows)]
        high_confidence = [15.0 + (i % 20) for i in range(n_rows)]
        data_value_types_list = [data_value_types[i % len(data_value_types)] for i in range(n_rows)]
        geo_locations = [f'({30 + i % 20}, {-120 + i % 30})' for i in range(n_rows)]
        strat1 = [stratifications[i % len(stratifications)] for i in range(n_rows)]
        strat_cat1 = [strat_categories[i % len(strat_categories)] for i in range(n_rows)]
        response_ids = [f'R{i:04d}' for i in range(n_rows)]
        location_ids = [i % 50 for i in range(n_rows)]
        
        # Verify all lists have the same length
        assert len(year_starts) == n_rows
        assert len(location_abbrs) == n_rows
        assert len(location_descs) == n_rows
        assert len(topics_list) == n_rows
        assert len(questions_list) == n_rows
        assert len(data_values) == n_rows
        assert len(low_confidence) == n_rows
        assert len(high_confidence) == n_rows
        assert len(data_value_types_list) == n_rows
        assert len(geo_locations) == n_rows
        assert len(strat1) == n_rows
        assert len(strat_cat1) == n_rows
        assert len(response_ids) == n_rows
        assert len(location_ids) == n_rows
        
        data = {
            'YearStart': year_starts,
            'LocationAbbr': location_abbrs,
            'LocationDesc': location_descs,
            'Topic': topics_list,
            'Question': questions_list,
            'DataValue': data_values,
            'LowConfidenceLimit': low_confidence,
            'HighConfidenceLimit': high_confidence,
            'DataValueType': data_value_types_list,
            'GeoLocation': geo_locations,
            'Stratification1': strat1,
            'StratificationCategory1': strat_cat1,
            'ResponseID': response_ids,
            'LocationID': location_ids
        }
        df = pd.DataFrame(data)
        
        # Verify DataFrame has correct shape
        assert df.shape[0] == n_rows
        assert df.shape[1] == len(data)

        # Save to temp file
        file_path = tmp_path / "test_data.csv"
        df.to_csv(file_path, index=False)

        # Run pipeline
        from src.ingestion.data_loading import load_data
        from src.ingestion.validator import run_all_validations
        from src.preprocessing.data_cleaning import clean_data
        from src.preprocessing.feature_engineering import engineer_features
        from src.models.trainer import ModelTrainer, RandomForestModel

        # 1. Load
        loaded_df = load_data(file_path)
        assert len(loaded_df) == n_rows
        logger.info(f"Loaded: {len(loaded_df)} rows")

        # 2. Validate - add YearEnd for validation
        loaded_df['YearEnd'] = loaded_df['YearStart']
        run_all_validations(loaded_df)

        # 3. Clean
        cleaned = clean_data(loaded_df)
        assert cleaned.shape[0] > 10, f"Too few rows after cleaning: {cleaned.shape[0]}"
        logger.info(f"After cleaning: {cleaned.shape[0]} rows")

        # 4. Feature engineering
        engineered = engineer_features(cleaned)
        assert 'DataValue' in engineered.columns
        assert len(engineered) > 10, f"Too few rows after feature engineering: {len(engineered)}"
        logger.info(f"After feature engineering: {len(engineered)} rows, {engineered.shape[1]} columns")

        # 5. Drop leakage columns BEFORE splitting
        leakage_cols = [
            'LowConfidenceLimit', 'HighConfidenceLimit', 
            'ConfidenceIntervalWidth', 'DataValueType',
            'DataValue_Missing_Flag', 'DataValueAlt',
            'DataValueUnit', 'DataValueFootnoteSymbol'
        ]
        for col in leakage_cols:
            if col in engineered.columns:
                engineered = engineered.drop(columns=[col])
                logger.info(f"Dropped leakage column: {col}")

        # Drop ID columns that might leak
        id_cols = [col for col in engineered.columns if 'ID' in col]
        for col in id_cols:
            if col in engineered.columns:
                engineered = engineered.drop(columns=[col])
                logger.info(f"Dropped ID column: {col}")

        # Ensure we have enough data for splitting (at least 20 rows)
        assert len(engineered) >= 20, f"Not enough data after processing: {len(engineered)} rows"

        # 6. Model training with smaller test_size for small dataset
        config = {'n_estimators': 10, 'random_state': 42}
        model = RandomForestModel(config=config)
        trainer = ModelTrainer(
            model=model, 
            test_size=0.15,  # Smaller test size
            val_size=0.1,
            do_hyperparameter_tuning=False,
            do_cross_validation=False,
            random_state=42
        )
        trainer.split_data(engineered, target_col='DataValue')
        trainer.train_with_cv(cv_folds=2)
        metrics = trainer.evaluate()

        assert 'R2' in metrics
        assert metrics['R2'] is not None
        logger.info(f"Test complete. R2: {metrics['R2']}")


    def test_end_to_end_pipeline_with_small_dataset(self, tmp_path):
        """Test that the pipeline handles small datasets gracefully."""
        # Create minimal dataset (20 rows)
        n_rows = 20
        
        data = {
            'YearStart': [2020] * n_rows,
            'LocationAbbr': ['CA'] * n_rows,
            'LocationDesc': ['California'] * n_rows,
            'Topic': ['Diabetes'] * n_rows,
            'Question': ['Prevalence'] * n_rows,
            'DataValue': [str(10 + i) for i in range(n_rows)],
            'LowConfidenceLimit': [5.0 + i for i in range(n_rows)],
            'HighConfidenceLimit': [15.0 + i for i in range(n_rows)],
            'DataValueType': ['Prevalence'] * n_rows,
            'GeoLocation': ['(36.0, -115.0)'] * n_rows,
            'Stratification1': ['Overall'] * n_rows,
            'StratificationCategory1': ['Overall'] * n_rows,
            'ResponseID': [f'R{i:04d}' for i in range(n_rows)],
            'LocationID': [i for i in range(n_rows)]
        }
        df = pd.DataFrame(data)
        
        # Verify DataFrame has correct shape
        assert df.shape[0] == n_rows
        assert df.shape[1] == len(data)

        # Save to temp file
        file_path = tmp_path / "small_test_data.csv"
        df.to_csv(file_path, index=False)

        # Run pipeline
        from src.ingestion.data_loading import load_data
        from src.ingestion.validator import run_all_validations
        from src.preprocessing.data_cleaning import clean_data
        from src.preprocessing.feature_engineering import engineer_features
        from src.models.trainer import ModelTrainer, RandomForestModel

        # 1. Load
        loaded_df = load_data(file_path)
        assert len(loaded_df) == n_rows

        # 2. Validate - add YearEnd for validation
        loaded_df['YearEnd'] = loaded_df['YearStart']
        run_all_validations(loaded_df)

        # 3. Clean
        cleaned = clean_data(loaded_df)
        assert cleaned.shape[0] > 0

        # 4. Feature engineering
        engineered = engineer_features(cleaned)
        assert 'DataValue' in engineered.columns

        # 5. Drop leakage columns
        leakage_cols = ['LowConfidenceLimit', 'HighConfidenceLimit', 'DataValueType']
        for col in leakage_cols:
            if col in engineered.columns:
                engineered = engineered.drop(columns=[col])

        # 6. Model training with very small test_size
        config = {'n_estimators': 5, 'random_state': 42}
        model = RandomForestModel(config=config)
        trainer = ModelTrainer(
            model=model, 
            test_size=0.1,
            val_size=0.1,
            do_hyperparameter_tuning=False,
            do_cross_validation=False,
            random_state=42
        )
        trainer.split_data(engineered, target_col='DataValue')
        trainer.train_with_cv(cv_folds=2)
        metrics = trainer.evaluate()

        assert 'R2' in metrics
        logger.info(f"Small dataset test complete. R2: {metrics['R2']}")