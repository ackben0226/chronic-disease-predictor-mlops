import pytest
import pandas as pd
import numpy as np
from src.preprocessing.feature_engineering import engineer_features

class TestFeatureEngineering:
    
    def test_engineer_features_creates_missing_flag(self):
        """Test that DataValue_Missing_Flag is created."""
        df = pd.DataFrame({
            'DataValue': [10, np.nan, 30],
            'DataValueType': ['A', 'A', 'A']
        })
        engineered = engineer_features(df)
        assert 'DataValue_Missing_Flag' in engineered.columns
        assert engineered['DataValue_Missing_Flag'].sum() == 1
    
    def test_engineer_features_creates_lat_long(self):
        """Test that Latitude and Longitude are extracted."""
        df = pd.DataFrame({
            'GeoLocation': ['(36.0, -115.0)', '(40.0, -80.0)'],
            'LocationAbbr': ['NV', 'PA']
        })
        engineered = engineer_features(df)
        assert 'Latitude' in engineered.columns
        assert 'Longitude' in engineered.columns
        assert engineered['Latitude'].iloc[0] == 36.0
    
    def test_engineer_features_creates_confidence_interval_width(self):
        """Test that ConfidenceIntervalWidth is created."""
        df = pd.DataFrame({
            'HighConfidenceLimit': [20.0, 30.0],
            'LowConfidenceLimit': [10.0, 20.0]
        })
        engineered = engineer_features(df)
        assert 'ConfidenceIntervalWidth' in engineered.columns
        assert engineered['ConfidenceIntervalWidth'].iloc[0] == 10.0
    
    def test_engineer_features_creates_region_features(self):
        """Test that US regions are created."""
        df = pd.DataFrame({
            'GeoLocation': ['(36.0, -115.0)'],
            'LocationAbbr': ['NV']
        })
        engineered = engineer_features(df)
        region_cols = [col for col in engineered.columns if col.startswith('Region_')]
        assert len(region_cols) > 0
    
    def test_engineer_features_creates_demographic_features(self):
        """Test that demographic one-hot encoding works."""
        df = pd.DataFrame({
            'Stratification1': ['Overall', 'Male', 'Female'],
            'GeoLocation': ['(36.0, -115.0)'] * 3,
            'LocationAbbr': ['NV'] * 3
        })
        engineered = engineer_features(df)
        demo_cols = [col for col in engineered.columns if col.startswith('Demo_')]
        assert len(demo_cols) >= 3
    
    def test_engineer_features_creates_time_features(self):
        """Test that YearsSince2020 is created."""
        df = pd.DataFrame({
            'YearStart': [2020, 2021, 2022],
            'GeoLocation': ['(36.0, -115.0)'] * 3,
            'LocationAbbr': ['NV'] * 3
        })
        engineered = engineer_features(df)
        assert 'YearsSince2020' in engineered.columns
        assert engineered['YearsSince2020'].iloc[0] == 0
        assert engineered['YearsSince2020'].iloc[2] == 2
    
    def test_engineer_features_returns_only_numeric(self):
        """Test that all returned columns are numeric."""
        df = pd.DataFrame({
            'DataValue': [10, 20, 30],
            'DataValueType': ['A', 'B', 'C'],
            'GeoLocation': ['(36.0, -115.0)'] * 3,
            'LocationAbbr': ['NV'] * 3,
            'Stratification1': ['Overall'] * 3,
            'YearStart': [2020] * 3
        })
        engineered = engineer_features(df)
        non_numeric = engineered.select_dtypes(include=['object', 'category']).columns
        assert len(non_numeric) == 0