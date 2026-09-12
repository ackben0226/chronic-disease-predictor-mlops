"""
monitoring/drift_detector.py
Detect data drift in model features using statistical methods.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DriftDetector:
    """
    Detect drift in feature distributions.
    
    Attributes:
        reference_stats_path: Path to the reference statistics JSON file
        threshold: Number of standard deviations for drift detection
        reference_stats: Cached reference statistics (loaded on init)
        created_at: Timestamp when detector was initialized
    """
    
    reference_stats_path: str = "models/training_data_stats.json"
    threshold: float = 2.0
    reference_stats: Dict[str, Any] = field(default_factory=dict, init=False)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(), init=False)
    
    def __post_init__(self):
        """Load reference statistics after initialization."""
        self.reference_stats = self._load_reference_stats()
        logger.info(f"🔍 DriftDetector initialized (threshold={self.threshold})")
    
    def _load_reference_stats(self) -> Dict[str, Any]:
        """
        Load reference statistics from file or create defaults.
        
        Returns:
            Dict containing reference statistics for features.
        """
        if Path(self.reference_stats_path).exists():
            try:
                with open(self.reference_stats_path, 'r') as f:
                    stats = json.load(f)
                logger.info(f"✅ Loaded reference stats from {self.reference_stats_path}")
                return stats
            except Exception as e:
                logger.warning(f"⚠️ Failed to load reference stats: {e}")
                return self._create_default_stats()
        else:
            logger.warning(f"⚠️ No reference stats found at {self.reference_stats_path}")
            return self._create_default_stats()
    
    def _create_default_stats(self) -> Dict[str, Any]:
        """
        Create default reference statistics.
        
        Returns:
            Default statistics dictionary.
        """
        logger.info("📊 Creating default reference statistics")
        return {
            "features": {
                "YearStart": {"mean": 2020.0, "std": 2.0},
                "Latitude": {"mean": 36.5, "std": 5.2},
                "Longitude": {"mean": -95.0, "std": 8.5},
            },
            "created_at": datetime.now().isoformat(),
            "source": "default"
        }
    
    def save_reference_stats(self, df: pd.DataFrame) -> None:
        """
        Save reference statistics from a DataFrame.
        
        Args:
            df: Training DataFrame to extract statistics from.
        """
        stats = {
            "features": {},
            "created_at": datetime.now().isoformat(),
            "source": "training_data",
            "n_samples": len(df)
        }
        
        # Only include numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            stats["features"][col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "q25": float(df[col].quantile(0.25)),
                "q50": float(df[col].median()),
                "q75": float(df[col].quantile(0.75))
            }
        
        # Save to file
        Path(self.reference_stats_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.reference_stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.reference_stats = stats
        logger.info(f"✅ Saved reference stats to {self.reference_stats_path} ({len(numeric_cols)} features)")
    
    def check_drift(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Check for drift in features.
        
        Args:
            X: Input DataFrame to check for drift.
        
        Returns:
            Dict containing drift report.
        """
        drift_report = {
            "timestamp": datetime.now().isoformat(),
            "drift_detected": False,
            "features": {},
            "summary": {
                "total_features": 0,
                "features_with_drift": 0,
                "features_missing": 0,
                "features_stable": 0
            }
        }
        
        # Only check numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        for feature in numeric_cols:
            drift_report["summary"]["total_features"] += 1
            
            # Check if feature exists in reference
            if feature not in self.reference_stats.get("features", {}):
                drift_report["features"][feature] = {
                    "status": "unknown",
                    "message": f"Feature '{feature}' not in reference stats"
                }
                drift_report["summary"]["features_missing"] += 1
                continue
            
            # Get reference stats
            ref = self.reference_stats["features"][feature]
            ref_mean = ref.get("mean", 0)
            ref_std = ref.get("std", 1)
            
            # Calculate current stats
            current_mean = X[feature].mean()
            current_std = X[feature].std()
            n_samples = len(X[feature].dropna())
            
            # Calculate deviation
            if ref_std > 0:
                deviation = abs(current_mean - ref_mean) / ref_std
            else:
                deviation = 0
            
            # Check for drift
            has_drift = deviation > self.threshold
            
            # Build feature report
            feature_report = {
                "status": "drift_detected" if has_drift else "stable",
                "ref_mean": float(ref_mean),
                "current_mean": float(current_mean),
                "ref_std": float(ref_std),
                "current_std": float(current_std),
                "deviation": float(deviation),
                "n_samples": n_samples
            }
            
            if has_drift:
                feature_report["message"] = (
                    f"Mean shifted from {ref_mean:.2f} to {current_mean:.2f} "
                    f"({deviation:.2f} std deviations)"
                )
                drift_report["summary"]["features_with_drift"] += 1
                drift_report["drift_detected"] = True
            else:
                drift_report["summary"]["features_stable"] += 1
            
            drift_report["features"][feature] = feature_report
        
        # Log summary
        if drift_report["drift_detected"]:
            logger.warning(
                f"⚠️ Drift detected: {drift_report['summary']['features_with_drift']} "
                f"out of {drift_report['summary']['total_features']} features"
            )
        else:
            logger.info(f"✅ No drift detected ({drift_report['summary']['features_stable']} features stable)")
        
        return drift_report
    
    def detect_outliers(self, X: pd.DataFrame, iqr_multiplier: float = 3.0) -> Dict[str, Any]:
        """
        Detect outliers using IQR method.
        
        Args:
            X: Input DataFrame
            iqr_multiplier: IQR multiplier for outlier detection (default: 3.0)
        
        Returns:
            Dict containing outlier report.
        """
        outlier_report = {
            "timestamp": datetime.now().isoformat(),
            "total_outliers": 0,
            "total_rows": len(X),
            "features": {}
        }
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        for feature in numeric_cols:
            series = X[feature].dropna()
            if len(series) == 0:
                continue
            
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR == 0:
                continue
            
            lower_bound = Q1 - iqr_multiplier * IQR
            upper_bound = Q3 + iqr_multiplier * IQR
            
            outliers = (series < lower_bound) | (series > upper_bound)
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                outlier_report["features"][feature] = {
                    "outlier_count": int(outlier_count),
                    "outlier_percent": float(outlier_count / len(series) * 100),
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                    "min_value": float(series.min()),
                    "max_value": float(series.max())
                }
                outlier_report["total_outliers"] += outlier_count
        
        if outlier_report["total_outliers"] > 0:
            logger.warning(f"⚠️ Found {outlier_report['total_outliers']} outliers across {len(outlier_report['features'])} features")
        else:
            logger.info("✅ No outliers detected")
        
        return outlier_report
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current drift detector state.
        
        Returns:
            Dict with detector summary.
        """
        return {
            "reference_stats_path": self.reference_stats_path,
            "threshold": self.threshold,
            "created_at": self.created_at,
            "features_monitored": len(self.reference_stats.get("features", {})),
            "reference_created_at": self.reference_stats.get("created_at", "unknown"),
            "reference_source": self.reference_stats.get("source", "unknown")
        }


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def create_drift_detector(
    reference_stats_path: str = "models/training_data_stats.json",
    threshold: float = 2.0
) -> DriftDetector:
    """
    Create a DriftDetector instance.
    
    Args:
        reference_stats_path: Path to reference statistics file.
        threshold: Drift detection threshold in standard deviations.
    
    Returns:
        DriftDetector instance.
    """
    return DriftDetector(
        reference_stats_path=reference_stats_path,
        threshold=threshold
    )