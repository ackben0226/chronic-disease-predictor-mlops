import pytest
import pandas as pd
import numpy as np
from src.ingestion.validator import (
    validate_schema,
    validate_not_empty,
    validate_value_ranges,
    validate_basic_quality,
    run_raw_data_validations
)

class TestValidator:
    
    def test_validate_schema_passes_with_valid_columns(self):
        """Test that schema validation passes with all required columns."""
        df = pd.DataFrame({
            'YearStart': [2020],
            'LocationAbbr': ['CA'],
            'DataValue': [15.5]
        })
        required = ['YearStart', 'LocationAbbr', 'DataValue']
        validate_schema(df, required)  # Should not raise
    
    def test_validate_schema_raises_with_missing_columns(self):
        """Test that schema validation raises error for missing columns."""
        df = pd.DataFrame({'YearStart': [2020]})
        required = ['YearStart', 'LocationAbbr', 'DataValue']
        with pytest.raises(ValueError) as exc:
            validate_schema(df, required)
        assert 'Missing required columns' in str(exc.value)
    
    def test_validate_not_empty_passes_with_data(self):
        """Test that empty validation passes with non-empty DataFrame."""
        df = pd.DataFrame({'col': [1, 2, 3]})
        validate_not_empty(df)  # Should not raise
    
    def test_validate_not_empty_raises_with_empty_dataframe(self):
        """Test that empty validation raises error for empty DataFrame."""
        df = pd.DataFrame()
        with pytest.raises(ValueError) as exc:
            validate_not_empty(df)
        assert 'empty' in str(exc.value)
    
    def test_validate_value_ranges_passes_valid_data(self):
        """Test that range validation passes for values within bounds."""
        df = pd.DataFrame({'YearStart': [2020, 2021, 2022]})
        validate_value_ranges(df, {'YearStart': (2000, 2030)})
    
    # tests/test_validator.py
    def test_validate_value_ranges_raises_out_of_bounds(self):
        """Test that range validation raises for out-of-bounds values."""
        df = pd.DataFrame({'YearStart': [1800, 2020]})
        with pytest.raises(ValueError) as exc:
            validate_value_ranges(df, {'YearStart': (2000, 2030)})
        # Check for the new error message format
        assert 'outside' in str(exc.value)
        assert '2000' in str(exc.value)
        assert '2030' in str(exc.value)
    
    def test_validate_basic_quality_catches_extreme_years(self):
        """Test that basic quality check warns about extreme years."""
        df = pd.DataFrame({'YearStart': [1800, 2020, 2021]})
        # Should log warning but not raise
        validate_basic_quality(df)
    
    def test_run_all_validations_passes_on_valid_data(self):
        """Test that full validation pipeline passes."""
        df = pd.DataFrame({
            'YearStart': [2020, 2021],
            'LocationAbbr': ['CA', 'NY'],
            'LocationDesc': ['California', 'New York'],
            'Topic': ['Diabetes', 'Cancer'],
            'Question': ['Prevalence', 'Rate'],
            'DataValue': ['15.5', '20.3'],
            'LowConfidenceLimit': [10.0, 15.0],
            'HighConfidenceLimit': [20.0, 25.0],
            'DataValueType': ['Prevalence', 'Rate']
        })
        run_raw_data_validations(df)  # Should not raise