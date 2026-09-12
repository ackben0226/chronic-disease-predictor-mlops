import pytest
import pandas as pd
from pathlib import Path
from src.ingestion.data_loading import load_data

class TestDataLoading:
    
    def test_load_data_returns_dataframe(self):
        """Test that load_data returns a pandas DataFrame."""
        df = load_data()
        assert isinstance(df, pd.DataFrame)
    
    def test_load_data_has_expected_columns(self):
        """Test that loaded data contains all expected columns."""
        required_cols = ['YearStart', 'LocationAbbr', 'DataValue']
        df = load_data()
        for col in required_cols:
            assert col in df.columns
    
    def test_load_data_not_empty(self):
        """Test that loaded data has at least one row."""
        df = load_data()
        assert len(df) > 0
    
    # tests/test_data_loading.py
    def test_load_data_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        from pathlib import Path
        with pytest.raises(FileNotFoundError):
            load_data(Path("nonexistent.csv"))  # Pass Path object, not string