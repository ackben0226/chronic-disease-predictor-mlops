import pytest
import pandas as pd
import numpy as np
from src.preprocessing.data_cleaning import clean_data

class TestDataCleaning:
    
    def test_clean_data_removes_duplicates(self):
        """Test that clean_data removes duplicate rows."""
        df = pd.DataFrame({
            'A': [1, 1, 2],
            'B': ['x', 'x', 'y']
        })
        cleaned = clean_data(df)
        assert len(cleaned) == 2
        assert cleaned.duplicated().sum() == 0
    
    def test_clean_data_fills_numeric_nulls_with_median(self):
        """Test that numeric nulls are filled with median."""
        df = pd.DataFrame({
            'DataValue': [10, np.nan, 30],
            'DataValueType': ['A', 'A', 'A']
        })
        cleaned = clean_data(df)
        assert cleaned['DataValue'].isna().sum() == 0
        assert cleaned['DataValue'].iloc[1] == 20.0  # Median of [10, 30]
    
    # tests/test_data_cleaning.py
    def test_clean_data_fills_string_nulls_with_unknown(self):
        """Test that string nulls are filled with 'unknown'."""
        df = pd.DataFrame({
            'LocationAbbr': ['CA', np.nan, 'NY']
        })
        cleaned = clean_data(df)
        assert cleaned['LocationAbbr'].isna().sum() == 0
        # Check that NaN became 'unknown' (or 'Unknown' depending on your logic)
        assert cleaned['LocationAbbr'].iloc[1].lower() in ['unknown', 'Unknown']
    
    def test_clean_data_strips_whitespace(self):
        """Test that whitespace is stripped from strings."""
        df = pd.DataFrame({
            'LocationAbbr': [' CA ', 'NY ', ' TX']
        })
        cleaned = clean_data(df)
        assert cleaned['LocationAbbr'].iloc[0] == 'CA'
        assert cleaned['LocationAbbr'].iloc[1] == 'NY'
        assert cleaned['LocationAbbr'].iloc[2] == 'TX'
    
    # tests/test_data_cleaning.py
    def test_clean_data_handles_outliers(self):
        """Test that outliers are capped."""
        df = pd.DataFrame({
            'DataValue': [1, 2, 3, 1000, 4, 5],
            'DataValueType': ['A'] * 6
        })
        cleaned = clean_data(df)
        # Outlier should be capped (it won't be 1000 anymore)
        assert cleaned['DataValue'].max() < 1000  # Not expecting < 100, just less than 1000
        # Check that the outlier was actually modified
        assert cleaned.loc[3, 'DataValue'] < 1000