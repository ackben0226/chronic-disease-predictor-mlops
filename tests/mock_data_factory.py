"""
tests/mock_data_factory.py
Factory for creating mock CDC Chronic Disease Indicator datasets.
Provides realistic synthetic data for testing.

Usage:
    from tests.mock_data_factory import MockDataFactory
    
    df = MockDataFactory.create_full_dataset(n_rows=100)
    df_with_missing = MockDataFactory.create_with_missing_values(n_rows=100)
    df_with_outliers = MockDataFactory.create_with_outliers(n_rows=100)
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from pathlib import Path
import tempfile


class MockDataFactory:
    """
    Factory class for creating mock CDC dataset variations.
    All methods use fixed seeds for reproducibility.
    """
    
    # ============================================================
    # CONSTANTS (Match real CDC dataset patterns)
    # ============================================================
    
    # US States and Territories
    STATES = [
        'CA', 'NY', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI',
        'NJ', 'VA', 'WA', 'AZ', 'MA', 'TN', 'IN', 'MO', 'MD', 'WI',
        'CO', 'MN', 'SC', 'AL', 'LA', 'KY', 'OR', 'OK', 'CT', 'UT',
        'IA', 'NV', 'AR', 'MS', 'KS', 'NM', 'NE', 'WV', 'ID', 'HI',
        'NH', 'ME', 'MT', 'RI', 'DE', 'SD', 'ND', 'AK', 'DC', 'VT',
        'WY', 'PR', 'VI', 'GU', 'AS', 'MP'
    ]
    
    TOPICS = [
        'Diabetes', 'Cancer', 'Heart Disease', 'Stroke', 'Obesity',
        'Smoking', 'Hypertension', 'Asthma', 'Arthritis', 'Depression'
    ]
    
    QUESTIONS = [
        'Prevalence', 'Rate', 'Mortality', 'Incidence', 'Hospitalization',
        'Emergency Visits', 'Cost', 'Disability', 'Quality of Life'
    ]
    
    DATA_VALUE_TYPES = [
        'Crude Prevalence', 'Age-adjusted Prevalence', 'Crude Rate',
        'Age-adjusted Rate', 'Number', 'Mean', 'Median', 'Percent'
    ]
    
    STRATIFICATIONS = [
        'Overall', 'Male', 'Female', 'White, non-Hispanic', 
        'Black, non-Hispanic', 'Hispanic', 'Asian, non-Hispanic',
        'Age 18-24', 'Age 25-44', 'Age 45-64', 'Age 65+'
    ]
    
    STRATIFICATION_CATEGORIES = [
        'Overall', 'Gender', 'Race/Ethnicity', 'Age'
    ]
    
    # State centroids for GeoLocation
    STATE_CENTROIDS = {
        'CA': (36.1162, -119.6816), 'NY': (42.1657, -74.9481),
        'TX': (31.0545, -97.5635), 'FL': (27.7663, -81.6868),
        'IL': (40.3495, -88.9861), 'PA': (40.5908, -77.2098),
        'OH': (40.3888, -82.7649), 'GA': (32.3294, -83.1137),
        'NC': (35.6301, -79.8064), 'MI': (43.3266, -84.5361),
        'NJ': (40.2989, -74.5210), 'VA': (37.7693, -78.1693),
        'WA': (47.4009, -121.4905), 'AZ': (33.7298, -111.4312),
        'MA': (42.2302, -71.5301), 'TN': (35.7478, -86.6923),
        'IN': (39.8934, -86.1346), 'MO': (38.4561, -92.2884),
        'MD': (39.0639, -76.8021), 'WI': (44.2685, -89.6165),
        'CO': (39.0598, -105.3111), 'MN': (45.6945, -93.9002),
        'SC': (33.8569, -80.9450), 'AL': (32.8067, -86.7911),
        'LA': (31.1695, -91.8678), 'KY': (37.6681, -84.6701),
        'OR': (44.5720, -122.0709), 'OK': (35.5653, -96.9289),
        'CT': (41.5978, -72.7554), 'UT': (40.1500, -111.8624),
        'IA': (42.0115, -93.2105), 'NV': (38.3135, -117.0558),
        'AR': (34.9697, -92.3731), 'MS': (32.7416, -89.6787),
        'KS': (38.5266, -96.7265), 'NM': (34.8405, -106.2485),
        'NE': (41.1254, -98.2681), 'WV': (38.4912, -80.9545),
        'ID': (44.2405, -114.4788), 'HI': (21.0943, -157.4983),
        'NH': (43.4525, -71.5639), 'ME': (44.6939, -69.3819),
        'MT': (46.9219, -110.4544), 'RI': (41.6809, -71.5118),
        'DE': (38.9108, -75.5277), 'SD': (44.2998, -99.4388),
        'ND': (47.5289, -99.7840), 'AK': (61.3707, -152.4044),
        'DC': (38.9072, -77.0369), 'VT': (44.0459, -72.7107),
        'WY': (42.7560, -107.3025), 'PR': (18.2208, -66.5901),
        'VI': (18.3358, -64.8963), 'GU': (13.4443, 144.7937),
        'AS': (-14.2710, -170.1322), 'MP': (15.0979, 145.6739)
    }
    
    # ============================================================
    # CORE CREATION METHODS
    # ============================================================
    
    @classmethod
    def create_full_dataset(
        cls,
        n_rows: int = 100,
        seed: int = 42,
        include_all_columns: bool = True
    ) -> pd.DataFrame:
        """
        Create a complete mock CDC dataset.
        
        Args:
            n_rows: Number of rows to generate
            seed: Random seed for reproducibility
            include_all_columns: If True, includes all real CDC columns
        
        Returns:
            DataFrame with mock data
        """
        np.random.seed(seed)
        
        # Helper: repeat list to exact length
        def repeat_to_length(arr, n):
            return (arr * (n // len(arr) + 1))[:n]
        
        # 1. Generate base values
        data_values = np.random.uniform(1, 80, n_rows)
        low_limits = np.maximum(0, data_values * 0.6 + np.random.uniform(-5, 5, n_rows))
        high_limits = data_values * 1.4 + np.random.uniform(-5, 5, n_rows)
        high_limits = np.maximum(low_limits + 0.5, high_limits)
        
        # 2. Generate state-based locations
        states = repeat_to_length(cls.STATES, n_rows)
        locations = repeat_to_length([f'State_{i}' for i in range(20)], n_rows)
        
        # 3. Generate GeoLocation strings
        geo_locations = []
        for state in states:
            lat, lon = cls.STATE_CENTROIDS.get(state, (39.8283, -98.5795))
            # Add some noise to make it realistic
            lat_noise = np.random.uniform(-0.5, 0.5)
            lon_noise = np.random.uniform(-0.5, 0.5)
            geo_locations.append(f'({lat + lat_noise:.2f}, {lon + lon_noise:.2f})')
        
        # 4. Build the DataFrame
        data = {
            'YearStart': np.random.randint(2015, 2025, n_rows),
            'YearEnd': np.random.randint(2015, 2025, n_rows),
            'LocationAbbr': states,
            'LocationDesc': locations,
            'Topic': repeat_to_length(cls.TOPICS, n_rows),
            'Question': repeat_to_length(cls.QUESTIONS, n_rows),
            'DataValue': [str(round(v, 1)) for v in data_values],
            'DataValueAlt': [round(v * 1.05, 1) for v in data_values],
            'DataValueUnit': repeat_to_length(['%', 'per 100,000', 'number'], n_rows),
            'DataValueType': repeat_to_length(cls.DATA_VALUE_TYPES, n_rows),
            'LowConfidenceLimit': [round(v, 1) for v in low_limits],
            'HighConfidenceLimit': [round(v, 1) for v in high_limits],
            'StratificationCategory1': repeat_to_length(cls.STRATIFICATION_CATEGORIES, n_rows),
            'Stratification1': repeat_to_length(cls.STRATIFICATIONS, n_rows),
            'StratificationCategory2': repeat_to_length(['Gender', 'Race/Ethnicity', 'Age', ''], n_rows),
            'Stratification2': repeat_to_length(['', 'Male', 'Female', 'White', 'Black'], n_rows),
            'GeoLocation': geo_locations,
            'ResponseID': [f'R{i:04d}' for i in range(n_rows)],
            'LocationID': np.random.randint(1, 100, n_rows),
            'TopicID': [f'T{i:03d}' for i in np.random.randint(1, 50, n_rows)],
            'QuestionID': [f'Q{i:03d}' for i in np.random.randint(1, 50, n_rows)],
            'DataValueTypeID': [f'DVT{i:03d}' for i in np.random.randint(1, 20, n_rows)],
            'StratificationCategoryID1': [f'SC{i:03d}' for i in np.random.randint(1, 10, n_rows)],
            'StratificationID1': [f'S{i:03d}' for i in np.random.randint(1, 20, n_rows)],
            'DataSource': repeat_to_length(['CDC', 'BRFSS', 'National Survey'], n_rows),
            'DataValueFootnoteSymbol': repeat_to_length(['', '*', '†', '‡'], n_rows),
            'DatavalueFootnote': repeat_to_length(['', 'Estimates suppressed', 'Data unreliable', ''], n_rows),
        }
        
        df = pd.DataFrame(data)
        
        # Ensure YearEnd >= YearStart
        df.loc[df['YearEnd'] < df['YearStart'], 'YearEnd'] = df['YearStart'] + np.random.randint(0, 2, len(df))
        
        return df
    
    # ============================================================
    # VARIANT CREATION METHODS
    # ============================================================
    
    @classmethod
    def create_with_missing_values(
        cls,
        n_rows: int = 100,
        missing_rate: float = 0.2,
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Create mock data with missing values.
        
        Args:
            n_rows: Number of rows
            missing_rate: Proportion of values to set to NaN (0.0 - 1.0)
            seed: Random seed
        """
        np.random.seed(seed)
        df = cls.create_full_dataset(n_rows, seed=seed)
        
        # Columns to inject missing values
        target_cols = ['DataValue', 'LocationAbbr', 'LowConfidenceLimit', 
                      'HighConfidenceLimit', 'GeoLocation']
        
        for col in target_cols:
            if col in df.columns:
                mask = np.random.random(n_rows) < missing_rate
                df.loc[mask, col] = np.nan
        
        return df
    
    @classmethod
    def create_with_outliers(
        cls,
        n_rows: int = 100,
        outlier_count: int = 5,
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Create mock data with extreme outliers.
        
        Args:
            n_rows: Number of rows
            outlier_count: Number of outlier rows to inject
            seed: Random seed
        """
        np.random.seed(seed)
        df = cls.create_full_dataset(n_rows, seed=seed)
        
        # Inject extreme outliers
        outlier_indices = np.random.choice(n_rows, size=min(outlier_count, n_rows), replace=False)
        df.loc[outlier_indices, 'DataValue'] = np.random.uniform(500, 3000, len(outlier_indices))
        df.loc[outlier_indices, 'DataValueType'] = 'Number'
        
        return df
    
    @classmethod
    def create_with_duplicates(
        cls,
        n_rows: int = 100,
        duplicate_rate: float = 0.1,
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Create mock data with duplicate rows.
        
        Args:
            n_rows: Number of rows
            duplicate_rate: Proportion of rows to duplicate
            seed: Random seed
        """
        np.random.seed(seed)
        df = cls.create_full_dataset(n_rows, seed=seed)
        
        # Create duplicates
        n_duplicates = int(n_rows * duplicate_rate)
        duplicate_indices = np.random.choice(n_rows, size=n_duplicates, replace=False)
        duplicate_rows = df.iloc[duplicate_indices].copy()
        
        # Append duplicates
        df = pd.concat([df, duplicate_rows], ignore_index=True)
        
        # Shuffle to mix duplicates
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        return df
    
    @classmethod
    def create_with_constant_columns(
        cls,
        n_rows: int = 100,
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Create mock data with constant columns (for testing warnings).
        
        Args:
            n_rows: Number of rows
            seed: Random seed
        """
        df = cls.create_full_dataset(n_rows, seed=seed)
        
        # Add constant columns
        df['Constant_Topic'] = 'Diabetes'
        df['Constant_Question'] = 'Prevalence'
        df['Constant_Year'] = 2020
        
        return df
    
    @classmethod
    def create_varied_dataset(
        cls,
        n_rows: int = 100,
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Create highly varied mock data (no constant columns).
        Good for model training tests.
        
        Args:
            n_rows: Number of rows
            seed: Random seed
        """
        np.random.seed(seed)
        df = cls.create_full_dataset(n_rows, seed=seed)
        
        # Ensure all important columns have variance
        df['DataValue'] = np.random.uniform(1, 80, n_rows)
        df['LowConfidenceLimit'] = df['DataValue'] * 0.7
        df['HighConfidenceLimit'] = df['DataValue'] * 1.3
        
        return df
    
    # ============================================================
    # FILE CREATION METHODS
    # ============================================================
    
    @classmethod
    def create_csv_file(
        cls,
        n_rows: int = 100,
        seed: int = 42,
        file_path: Optional[Path] = None
    ) -> Path:
        """
        Create a mock CSV file on disk.
        
        Args:
            n_rows: Number of rows
            seed: Random seed
            file_path: Custom file path (optional)
        
        Returns:
            Path to the created CSV file
        """
        if file_path is None:
            temp_dir = Path(tempfile.mkdtemp())
            file_path = temp_dir / "mock_cdc_data.csv"
        
        df = cls.create_full_dataset(n_rows, seed=seed)
        df.to_csv(file_path, index=False)
        return file_path
    
    @classmethod
    def create_parquet_file(
        cls,
        n_rows: int = 100,
        seed: int = 42,
        file_path: Optional[Path] = None
    ) -> Path:
        """
        Create a mock Parquet file on disk.
        
        Args:
            n_rows: Number of rows
            seed: Random seed
            file_path: Custom file path (optional)
        
        Returns:
            Path to the created Parquet file
        """
        if file_path is None:
            temp_dir = Path(tempfile.mkdtemp())
            file_path = temp_dir / "mock_cdc_data.parquet"
        
        df = cls.create_full_dataset(n_rows, seed=seed)
        df.to_parquet(file_path, index=False)
        return file_path
    
    # ============================================================
    # QUICK ACCESS METHODS
    # ============================================================
    
    @classmethod
    def quick_sample(cls, n_rows: int = 10, seed: int = 42) -> pd.DataFrame:
        """Quickly get a small sample for quick testing."""
        return cls.create_full_dataset(n_rows, seed=seed)
    
    @classmethod
    def quick_sample_with_nulls(cls, n_rows: int = 10) -> pd.DataFrame:
        """Quickly get a small sample with nulls."""
        return cls.create_with_missing_values(n_rows, missing_rate=0.3)
    
    @classmethod
    def quick_sample_with_outliers(cls, n_rows: int = 10) -> pd.DataFrame:
        """Quickly get a small sample with outliers."""
        return cls.create_with_outliers(n_rows, outlier_count=2)


# ============================================================
# CONVENIENCE FUNCTIONS (for quick access in tests)
# ============================================================

def mock_data(n_rows: int = 100, seed: int = 42) -> pd.DataFrame:
    """Quick function to get mock data."""
    return MockDataFactory.create_full_dataset(n_rows, seed=seed)

def mock_data_with_missing(n_rows: int = 100, rate: float = 0.2) -> pd.DataFrame:
    """Quick function to get mock data with missing values."""
    return MockDataFactory.create_with_missing_values(n_rows, missing_rate=rate)

def mock_data_with_outliers(n_rows: int = 100) -> pd.DataFrame:
    """Quick function to get mock data with outliers."""
    return MockDataFactory.create_with_outliers(n_rows)

def mock_data_with_duplicates(n_rows: int = 100) -> pd.DataFrame:
    """Quick function to get mock data with duplicates."""
    return MockDataFactory.create_with_duplicates(n_rows)