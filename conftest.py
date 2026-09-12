"""
Pytest configuration and shared fixtures.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Import the mock data factory
from tests.mock_data_factory import MockDataFactory


# ============================================================
# MOCK DATA FIXTURES
# ============================================================

@pytest.fixture
def mock_df_10():
    """Get 10 rows of mock data."""
    return MockDataFactory.quick_sample(n_rows=10)


@pytest.fixture
def mock_df_100():
    """Get 100 rows of mock data."""
    return MockDataFactory.create_full_dataset(n_rows=100)


@pytest.fixture
def mock_df_with_missing():
    """Get mock data with missing values."""
    return MockDataFactory.create_with_missing_values(n_rows=100, missing_rate=0.2)


@pytest.fixture
def mock_df_with_outliers():
    """Get mock data with outliers."""
    return MockDataFactory.create_with_outliers(n_rows=100, outlier_count=5)


@pytest.fixture
def mock_df_with_duplicates():
    """Get mock data with duplicates."""
    return MockDataFactory.create_with_duplicates(n_rows=100, duplicate_rate=0.1)


@pytest.fixture
def mock_csv_file(tmp_path):
    """Create a temporary mock CSV file."""
    return MockDataFactory.create_csv_file(n_rows=100, file_path=tmp_path / "mock_data.csv")


# ============================================================
# SAMPLE DATA FIXTURES (for backward compatibility)
# ============================================================

@pytest.fixture
def sample_data():
    """Sample CDC-like dataset for testing."""
    return MockDataFactory.quick_sample(n_rows=20, seed=42)


@pytest.fixture
def sample_csv_file(tmp_path, sample_data):
    """Create a temporary CSV file for testing."""
    file_path = tmp_path / "test_data.csv"
    sample_data.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def sample_data_with_nulls():
    """Create sample data with null values."""
    return MockDataFactory.create_with_missing_values(n_rows=20, missing_rate=0.2)


@pytest.fixture
def sample_data_outliers():
    """Create sample data with outliers."""
    return MockDataFactory.create_with_outliers(n_rows=20)


@pytest.fixture
def small_dataset_for_model():
    """Create a small dataset for model testing."""
    return MockDataFactory.create_varied_dataset(n_rows=100)