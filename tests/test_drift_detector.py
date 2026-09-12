"""
tests/test_drift_detector.py
Test the drift detection system.
"""

import pytest
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
import tempfile
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from monitoring.drift_detector import DriftDetector

# ============================================================
# LOGGING SETUP
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class TestDriftDetector:
    """Test suite for DriftDetector."""

    def setup_method(self):
        """Set up test data before each test."""
        np.random.seed(42)  # For reproducibility
        
        # Create training data (reference)
        self.train_df = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(10, 2, 1000),
            'feature3': np.random.uniform(0, 100, 1000),
            'YearStart': np.random.randint(2015, 2025, 1000),
            'Latitude': np.random.normal(36.5, 5.2, 1000),
            'Longitude': np.random.normal(-95.0, 8.5, 1000)
        })
        
        # Create test data (no drift)
        self.test_df = pd.DataFrame({
            'feature1': np.random.normal(0.1, 1, 100),
            'feature2': np.random.normal(10.2, 2, 100),
            'feature3': np.random.uniform(1, 99, 100),
            'YearStart': np.random.randint(2015, 2025, 100),
            'Latitude': np.random.normal(36.7, 5.0, 100),
            'Longitude': np.random.normal(-94.8, 8.3, 100)
        })
        
        # Create drifted test data
        self.drifted_df = pd.DataFrame({
            'feature1': np.random.normal(5.0, 1, 100),  # Drifted by 5 std!
            'feature2': np.random.normal(10.2, 2, 100),
            'feature3': np.random.uniform(1, 99, 100),
            'YearStart': np.random.randint(2015, 2025, 100),
            'Latitude': np.random.normal(36.7, 5.0, 100),
            'Longitude': np.random.normal(-94.8, 8.3, 100)
        })
        
        logger.info("✅ Test data setup complete")

    def test_drift_detector_initialization(self):
        """Test that DriftDetector initializes correctly."""
        detector = DriftDetector(threshold=2.0)
        
        assert detector is not None
        assert detector.threshold == 2.0
        assert detector.reference_stats is not None
        assert isinstance(detector.reference_stats, dict)
        logger.info("✅ DriftDetector initialized successfully")

    def test_save_reference_stats(self):
        """Test saving reference statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir) / "test_stats.json"
            
            detector = DriftDetector(reference_stats_path=str(temp_path))
            detector.save_reference_stats(self.train_df)
            
            assert temp_path.exists()
            
            # Verify content
            with open(temp_path, 'r') as f:
                stats = json.load(f)
            
            assert "features" in stats
            assert "created_at" in stats
            assert "feature1" in stats["features"]
            assert "mean" in stats["features"]["feature1"]
            logger.info(f"✅ Reference stats saved to {temp_path}")

    def test_load_reference_stats(self):
        """Test loading reference statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir) / "test_stats.json"
            
            # Save first
            detector1 = DriftDetector(reference_stats_path=str(temp_path))
            detector1.save_reference_stats(self.train_df)
            
            # Load second
            detector2 = DriftDetector(reference_stats_path=str(temp_path))
            
            assert detector2.reference_stats is not None
            assert len(detector2.reference_stats["features"]) > 0
            assert detector2.reference_stats["features"]["feature1"]["mean"] == pytest.approx(0, abs=0.1)
            logger.info("✅ Reference stats loaded successfully")

    def test_check_drift_no_drift(self):
        """Test drift detection on data with no drift."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir) / "test_stats.json"
            
            detector = DriftDetector(reference_stats_path=str(temp_path), threshold=2.0)
            detector.save_reference_stats(self.train_df)
            
            # Check for drift
            report = detector.check_drift(self.test_df)
            
            assert report["drift_detected"] is False
            assert report["summary"]["features_with_drift"] == 0
            assert report["summary"]["features_stable"] > 0
            logger.info(f"✅ No drift detected (features_stable: {report['summary']['features_stable']})")

    def test_check_drift_with_drift(self):
        """Test drift detection on data with drift."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir) / "test_stats.json"
            
            detector = DriftDetector(reference_stats_path=str(temp_path), threshold=2.0)
            detector.save_reference_stats(self.train_df)
            
            # Check for drift
            report = detector.check_drift(self.drifted_df)
            
            assert report["drift_detected"] is True
            assert report["summary"]["features_with_drift"] > 0
            
            # Check that feature1 detected drift
            assert "feature1" in report["features"]
            assert report["features"]["feature1"]["status"] == "drift_detected"
            logger.info(f"✅ Drift detected: {report['summary']['features_with_drift']} features drifted")

    def test_detect_outliers(self):
        """Test outlier detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir) / "test_stats.json"
            
            detector = DriftDetector(reference_stats_path=str(temp_path))
            detector.save_reference_stats(self.train_df)
            
            # Create data with outliers
            outlier_df = self.test_df.copy()
            outlier_df.loc[0, 'feature1'] = 50  # Extreme outlier
            outlier_df.loc[1, 'feature2'] = -50  # Extreme outlier
            
            # Detect outliers
            report = detector.detect_outliers(outlier_df, iqr_multiplier=3.0)
            
            assert report["total_outliers"] > 0
            assert "feature1" in report["features"]
            assert report["features"]["feature1"]["outlier_count"] > 0
            logger.info(f"✅ Outliers detected: {report['total_outliers']} total")

    def test_get_summary(self):
        """Test getting detector summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir) / "test_stats.json"
            
            detector = DriftDetector(reference_stats_path=str(temp_path), threshold=2.0)
            detector.save_reference_stats(self.train_df)
            
            summary = detector.get_summary()
            
            assert "reference_stats_path" in summary
            assert "threshold" in summary
            assert "features_monitored" in summary
            assert summary["features_monitored"] > 0
            logger.info(f"✅ Summary: monitoring {summary['features_monitored']} features")

    def test_realistic_scenario(self):
        """Test a realistic scenario with real-world data patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir) / "real_stats.json"
            
            # Create realistic training data (e.g., health data)
            np.random.seed(42)
            n_samples = 10000
            
            real_train = pd.DataFrame({
                'age': np.random.normal(45, 15, n_samples).clip(18, 90),
                'bmi': np.random.normal(28, 5, n_samples).clip(15, 50),
                'blood_pressure': np.random.normal(120, 15, n_samples).clip(90, 180),
                'diabetes_risk': np.random.normal(0.15, 0.08, n_samples).clip(0, 1)
            })
            
            # Realistic drifted data (e.g., patients getting older, higher BMI)
            real_drifted = pd.DataFrame({
                'age': np.random.normal(55, 15, 500).clip(18, 90),  # Older population
                'bmi': np.random.normal(32, 5, 500).clip(15, 50),   # Higher BMI
                'blood_pressure': np.random.normal(130, 15, 500).clip(90, 180),  # Higher BP
                'diabetes_risk': np.random.normal(0.25, 0.08, 500).clip(0, 1)  # Higher risk
            })
            
            detector = DriftDetector(reference_stats_path=str(temp_path), threshold=2.0)
            detector.save_reference_stats(real_train)
            
            report = detector.check_drift(real_drifted)
            
            logger.info("📊 Realistic Scenario Results:")
            logger.info(f"   Total features: {report['summary']['total_features']}")
            logger.info(f"   Features with drift: {report['summary']['features_with_drift']}")
            logger.info(f"   Features stable: {report['summary']['features_stable']}")
            
            # Should detect drift in at least one feature
            assert report["drift_detected"] is True
            logger.info("✅ Realistic scenario: drift detected as expected")


# ============================================================
# MANUAL TEST SCRIPT
# ============================================================

def test_drift_integration():
    """Integration test with the actual monitoring system."""
    logger.info("=" * 60)
    logger.info("DRIFT DETECTION INTEGRATION TEST")
    logger.info("=" * 60)
    
    # Use the actual drift detector
    from monitoring.drift_detector import DriftDetector
    
    # Create reference stats from a sample
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(10, 2, 1000)
    })
    
    # Save stats
    detector = DriftDetector(reference_stats_path="models/training_data_stats.json")
    detector.save_reference_stats(sample_data)
    logger.info("✅ Reference stats saved")
    
    # Test with similar data (no drift)
    no_drift = pd.DataFrame({
        'feature1': np.random.normal(0.1, 1, 100),
        'feature2': np.random.normal(10.1, 2, 100)
    })
    
    logger.info("🔍 Testing no drift scenario...")
    report = detector.check_drift(no_drift)
    logger.info(f"   Drift detected: {report['drift_detected']}")
    logger.info(f"   Features with drift: {report['summary']['features_with_drift']}")
    
    # Test with drifted data
    drift = pd.DataFrame({
        'feature1': np.random.normal(5, 1, 100),
        'feature2': np.random.normal(10.1, 2, 100)
    })
    
    logger.info("🔍 Testing drift scenario...")
    report = detector.check_drift(drift)
    logger.info(f"   Drift detected: {report['drift_detected']}")
    logger.info(f"   Features with drift: {report['summary']['features_with_drift']}")
    
    logger.info("=" * 60)
    logger.info("✅ Integration test complete")


# ============================================================
# RUN TESTS
# ============================================================

if __name__ == "__main__":
    # Configure logging for script execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Run integration test
    test_drift_integration()
    
    # Run pytest if available
    logger.info("\nRunning pytest tests...")
    logger.info("(Run 'pytest tests/test_drift_detector.py -v' for full test suite)")