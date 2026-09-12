"""
src/models/trainer.py
Handles data splitting, model training, evaluation, and saving.
Uses composition (has-a model) rather than inheritance for the pipeline.
"""

import pandas as pd
import logging
import numpy as np
import joblib
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from sklearn.model_selection import (
    train_test_split, 
    cross_val_score, 
    RandomizedSearchCV
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score,
    mean_absolute_percentage_error
)

# Import the base contract
from src.models.base import BaseModel

logger = logging.getLogger(__name__)


# ============================================================
# ENHANCED BASE MODEL WITH VALIDATION
# ============================================================

@dataclass
class RandomForestModel(BaseModel):
    """Wrapper for Sklearn's RandomForestRegressor with validation."""

    config: dict
    model: RandomForestRegressor = field(init=False)
    feature_names: Optional[List[str]] = field(default=None, init=False)
    is_trained: bool = field(default=False, init=False)
    training_history: Dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        """Instantiate the Sklearn model with config."""
        config_copy = self.config.copy()
        
        # Ensure reproducibility
        if 'random_state' not in config_copy:
            config_copy['random_state'] = 42
        
        # Set n_jobs to limit memory usage
        if 'n_jobs' not in config_copy:
            config_copy['n_jobs'] = 2
        
        self.model = RandomForestRegressor(**config_copy)
        self.training_history = {}

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[np.ndarray] = None):
        """Train the model with optional sample weights."""
        self.feature_names = X.columns.tolist()
        
        # Check for data quality before training
        self._validate_inputs(X, y)
        
        # Fit the model
        self.model.fit(X, y, sample_weight=sample_weight)
        self.is_trained = True
        
        logger.info(f"✅ RandomForest trained on {X.shape[0]:,} rows with {X.shape[1]} features.")
        
        # Store training info
        self.training_history['n_samples'] = X.shape[0]
        self.training_history['n_features'] = X.shape[1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model has not been trained yet.")
        
        # Ensure X has the right columns
        if self.feature_names is not None:
            missing_cols = set(self.feature_names) - set(X.columns)
            if missing_cols:
                raise ValueError(f"Missing columns: {missing_cols}")
        
        return self.model.predict(X)

    def predict_with_confidence(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with confidence intervals using tree variance.
        Returns: (predictions, std_devs)
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet.")
        
        # Get predictions from all trees
        tree_predictions = np.array([
            tree.predict(X.values) for tree in self.model.estimators_
        ])
        
        predictions = tree_predictions.mean(axis=0)
        std_devs = tree_predictions.std(axis=0)
        
        return predictions, std_devs

    def _validate_inputs(self, X: pd.DataFrame, y: pd.Series):
        """Validate inputs before training with proper type checking."""
        
        # --- 1. Ensure all columns are numeric ---
        non_numeric_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if non_numeric_cols:
            raise ValueError(
                f"Non-numeric columns found: {non_numeric_cols[:10]}...\n"
                f"Please ensure all features are numeric before training."
            )
        
        # --- 2. Check for infinite values (only on numeric columns) ---
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            if np.isinf(X[numeric_cols].values).any():
                raise ValueError("Infinite values found in X")
        
        # --- 3. Check for NaN ---
        if X.isnull().any().any():
            null_cols = X.columns[X.isnull().any()].tolist()
            null_counts = X[null_cols].isnull().sum().to_dict()
            raise ValueError(f"NaN values found in columns: {null_counts}")
        
        if y.isnull().any():
            raise ValueError(f"NaN values found in y: {y.isnull().sum()} rows")
        
        # --- 4. Check data types ---
        if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
            non_numeric = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
            raise ValueError(f"Not all columns are numeric: {non_numeric[:5]}...")

    def get_feature_importance(self) -> pd.DataFrame:
        """Extract feature importance."""
        if not self.is_trained:
            raise ValueError("Model not trained yet.")
        
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning("This model does not support feature importance.")
            return pd.DataFrame()
        
        importances = self.model.feature_importances_
        feature_names = self.feature_names
        
        if feature_names is None or len(feature_names) != len(importances):
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        df_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        return df_importance


# ============================================================
# THE PRODUCTION MODEL TRAINER
# ============================================================

class ModelTrainer:
    """
    Production-grade training pipeline with:
    - Cross-validation
    - Hyperparameter tuning
    - Data leakage prevention
    - Outlier detection
    - Feature importance analysis
    - Model registry
    """

    def __init__(
        self, 
        model: BaseModel, 
        test_size: float = 0.2, 
        val_size: float = 0.1,
        random_state: int = 42,
        do_hyperparameter_tuning: bool = True,
        do_cross_validation: bool = True
    ):
        self.model = model
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.do_hyperparameter_tuning = do_hyperparameter_tuning
        self.do_cross_validation = do_cross_validation
        
        # Data storage
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        # Metrics
        self.metrics = {}
        self.cv_scores = {}
        self.best_params = None
        
        # Track data leakage
        self.leakage_report = {}

    def split_data(
        self, 
        df: pd.DataFrame, 
        target_col: str,
        drop_cols: Optional[List[str]] = None
    ):
        """
        Split data into train/val/test sets with leakage prevention.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            drop_cols: Columns to drop (e.g., leakage columns)
        """
        logger.info("=" * 60)
        logger.info("SPLITTING DATA")
        logger.info("=" * 60)
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found!")
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # --- 1. Drop leakage columns ---
        if drop_cols:
            cols_to_drop = [col for col in drop_cols if col in df.columns]
            if cols_to_drop:
                logger.warning(f"⚠️ Dropping potential leakage columns: {cols_to_drop}")
                df = df.drop(columns=cols_to_drop)
                self.leakage_report['dropped_columns'] = cols_to_drop
        
        # --- 2. Check for target leakage ---
        self._check_target_leakage(df, target_col)
        
        # --- 3. Separate features and target ---
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # --- 4. Handle extreme outliers in target ---
        y, outlier_mask = self._cap_target_outliers(y)
        if outlier_mask is not None:
            X = X[~outlier_mask]
            y = y[~outlier_mask]
        
        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        
        # --- 5. Split into train + temp, then temp into val + test ---
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=self.test_size + self.val_size,
            random_state=self.random_state
        )
        
        val_size_adj = self.val_size / (self.test_size + self.val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=self.test_size / (self.test_size + self.val_size),
            random_state=self.random_state
        )
        
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        # --- 6. Log split info ---
        total = len(X)
        logger.info(f"Train: {len(X_train):,} rows ({len(X_train)/total*100:.1f}%)")
        logger.info(f"Validation: {len(X_val):,} rows ({len(X_val)/total*100:.1f}%)")
        logger.info(f"Test: {len(X_test):,} rows ({len(X_test)/total*100:.1f}%)")
        
        # --- 7. Validate split quality ---
        self._validate_split_quality()
        
        logger.info("=" * 60)

    def _check_target_leakage(self, df: pd.DataFrame, target_col: str):
        """
        Check for potential target leakage in features.
        """
        logger.info("Checking for potential data leakage...")
        
        leakage_warnings = []
        
        # 1. Check if target appears in features
        if target_col in df.columns:
            leakage_warnings.append(f"⚠️ Target column '{target_col}' is in features!")
        
        # 2. Check for columns with extremely high correlation (>0.95)
        for col in df.columns:
            if col != target_col:
                try:
                    corr = df[col].corr(df[target_col])
                    if abs(corr) > 0.95:
                        leakage_warnings.append(f"⚠️ '{col}' has correlation {corr:.3f} with target")
                except:
                    pass
        
        # 3. Check for ID columns
        id_cols = [col for col in df.columns if any(x in col.lower() for x in ['id', 'code', 'index'])]
        if id_cols:
            leakage_warnings.append(f"⚠️ ID columns found: {id_cols}")
        
        # 4. Check for future-looking columns
        if 'YearEnd' in df.columns and 'YearStart' in df.columns:
            future = df[df['YearEnd'] > df['YearStart']]
            if len(future) > 0:
                leakage_warnings.append(f"⚠️ 'YearEnd' > 'YearStart' for {len(future):,} rows")
        
        if leakage_warnings:
            logger.warning("=" * 50)
            logger.warning("⚠️ DATA LEAKAGE WARNINGS ⚠️")
            for warning in leakage_warnings:
                logger.warning(f"  {warning}")
            logger.warning("=" * 50)
            logger.warning("Consider dropping suspicious columns!")
            self.leakage_report['warnings'] = leakage_warnings
        else:
            logger.info("✅ No obvious data leakage detected.")

    def _cap_target_outliers(self, y: pd.Series) -> Tuple[pd.Series, Optional[pd.Series]]:
        """Cap extreme outliers in target to prevent model distortion."""
        Q1 = y.quantile(0.01)
        Q3 = y.quantile(0.99)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        outliers = (y < lower_bound) | (y > upper_bound)
        outlier_count = outliers.sum()
        
        if outlier_count > 0:
            outlier_pct = outlier_count / len(y) * 100
            logger.warning(
                f"⚠️ Target has {outlier_count:,} extreme outliers ({outlier_pct:.2f}%)"
            )
            logger.info(f"  Capping target at [{lower_bound:.2f}, {upper_bound:.2f}]")
            
            y_capped = y.clip(lower=lower_bound, upper=upper_bound)
            return y_capped, outliers
        else:
            return y, None

    def _validate_split_quality(self):
        """Validate that the split produced similar distributions."""
        y_train_mean = self.y_train.mean()
        y_val_mean = self.y_val.mean()
        y_test_mean = self.y_test.mean()
        
        logger.info("Checking split quality...")
        
        max_diff = max(
            abs(y_train_mean - y_val_mean) / max(abs(y_train_mean), 1e-10),
            abs(y_train_mean - y_test_mean) / max(abs(y_train_mean), 1e-10),
            abs(y_val_mean - y_test_mean) / max(abs(y_val_mean), 1e-10)
        ) * 100
        
        if max_diff > 20:
            logger.warning(
                f"⚠️ Target distribution differs across splits: "
                f"train={y_train_mean:.2f}, val={y_val_mean:.2f}, test={y_test_mean:.2f}"
            )
        else:
            logger.info(
                f"✅ Target distribution consistent: "
                f"train={y_train_mean:.2f}, val={y_val_mean:.2f}, test={y_test_mean:.2f}"
            )

    def train_with_cv(
        self, 
        cv_folds: int = 3,
        param_grid: Optional[Dict] = None
    ):
        """
        Train with cross-validation and optional hyperparameter tuning.
        """
        logger.info("=" * 60)
        logger.info("TRAINING WITH CROSS-VALIDATION")
        logger.info("=" * 60)
        
        if self.X_train is None:
            raise ValueError("Data has not been split yet. Call split_data() first.")
        
        # --- 1. Hyperparameter Tuning ---
        if self.do_hyperparameter_tuning and param_grid:
            logger.info("Performing hyperparameter tuning...")
            self._perform_hyperparameter_tuning(param_grid)
        
        # --- 2. Cross-Validation ---
        if self.do_cross_validation:
            logger.info(f"Performing {cv_folds}-fold cross-validation...")
            
            X_cv = pd.concat([self.X_train, self.X_val])
            y_cv = pd.concat([self.y_train, self.y_val])
            
            cv_scores = cross_val_score(
                self.model.model,
                X_cv, y_cv,
                cv=cv_folds,
                scoring='neg_mean_squared_error',
                n_jobs=1
            )
            
            self.cv_scores = {
                'mean': -cv_scores.mean(),
                'std': cv_scores.std(),
                'min': -cv_scores.min(),
                'max': -cv_scores.max(),
                'all': -cv_scores
            }
            
            logger.info(f"CV RMSE: {np.sqrt(self.cv_scores['mean']):.4f} ± {np.sqrt(self.cv_scores['std']):.4f}")
        
        # --- 3. Train on full training set ---
        logger.info("Training final model on full training set...")
        
        X_train_full = pd.concat([self.X_train, self.X_val])
        y_train_full = pd.concat([self.y_train, self.y_val])
        
        self.model.fit(X_train_full, y_train_full)
        
        logger.info("✅ Model training complete!")

    def _perform_hyperparameter_tuning(self, param_grid: Dict):
        """Perform hyperparameter tuning using validation set."""
        logger.info("Performing hyperparameter tuning...")
        
        X_tune = pd.concat([self.X_train, self.X_val])
        y_tune = pd.concat([self.y_train, self.y_val])
        
        model_tune = RandomForestRegressor(
            random_state=self.random_state, 
            n_jobs=2
        )
        
        random_search = RandomizedSearchCV(
            model_tune,
            param_distributions=param_grid,
            n_iter=15,
            cv=3,
            scoring='neg_mean_squared_error',
            random_state=self.random_state,
            n_jobs=1,
            error_score='raise'
        )
        
        logger.info(f"Trying {len(param_grid)} parameter combinations...")
        random_search.fit(X_tune, y_tune)
        
        self.best_params = random_search.best_params_
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best CV score: {-random_search.best_score_:.4f}")
        
        self.model.model.set_params(**self.best_params)

    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on test set with comprehensive metrics."""
        logger.info("=" * 60)
        logger.info("MODEL EVALUATION")
        logger.info("=" * 60)
        
        if self.X_test is None:
            raise ValueError("No test data available. Run split_data() first.")
        
        predictions = self.model.predict(self.X_test)
        
        mae = mean_absolute_error(self.y_test, predictions)
        mse = mean_squared_error(self.y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, predictions)
        mape = mean_absolute_percentage_error(self.y_test, predictions) * 100
        
        pred_min = predictions.min()
        pred_max = predictions.max()
        target_min = self.y_test.min()
        target_max = self.y_test.max()
        
        within_bounds = ((predictions >= target_min) & (predictions <= target_max)).sum()
        within_bounds_pct = within_bounds / len(predictions) * 100
        
        self.metrics = {
            'MAE': round(mae, 4),
            'MSE': round(mse, 4),
            'RMSE': round(rmse, 4),
            'R2': round(r2, 4),
            'MAPE (%)': round(mape, 2),
            'Prediction Range': f"{pred_min:.2f} - {pred_max:.2f}",
            'Target Range': f"{target_min:.2f} - {target_max:.2f}",
            'Within Bounds (%)': round(within_bounds_pct, 1)
        }
        
        logger.info("📊 EVALUATION METRICS:")
        for key, value in self.metrics.items():
            if key in ['Prediction Range', 'Target Range']:
                logger.info(f"  {key}: {value}")
            elif key == 'Within Bounds (%)':
                logger.info(f"  {key}: {value}%")
            else:
                logger.info(f"  {key}: {value}")
        
        if r2 > 0.99:
            logger.warning("⚠️ WARNING: R² > 0.99! Check for data leakage!")
        
        if within_bounds_pct < 90:
            logger.warning(
                f"⚠️ {100 - within_bounds_pct:.1f}% of predictions outside target range"
            )
        
        if self.cv_scores:
            cv_rmse = np.sqrt(self.cv_scores['mean'])
            if rmse < 0.8 * cv_rmse:
                logger.warning(
                    f"⚠️ Test RMSE ({rmse:.4f}) significantly lower than CV RMSE ({cv_rmse:.4f})"
                )
            elif rmse > 1.2 * cv_rmse:
                logger.warning(
                    f"⚠️ Test RMSE ({rmse:.4f}) significantly higher than CV RMSE ({cv_rmse:.4f})"
                )
            else:
                logger.info(f"✅ Test RMSE ({rmse:.4f}) consistent with CV ({cv_rmse:.4f})")
        
        logger.info("=" * 60)
        return self.metrics

    def save_model(self, path: Path, include_metadata: bool = True):
        """
        Save the trained model and metadata to disk.
        
        Uses standardized naming: chronic_disease_predictor_v1.0
        This ensures consistency between training and deployment.
        
        Args:
            path: Path where model should be saved (used for directory only)
            include_metadata: Whether to save metadata along with model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # ============================================================
        # STANDARDIZED NAMING - FIXED!
        # ============================================================
        model_name = "chronic_disease_predictor"
        version = "1.0"
        
        # 1. Save model with standardized name (using .joblib)
        model_path = path.parent / f"{model_name}_v{version}.joblib"
        joblib.dump(self.model, model_path)
        logger.info(f"✅ Model saved to: {model_path}")
        
        # 2. Save metadata with standardized name (using .metadata.json)
        if include_metadata:
            # Extract feature names from model if available
            feature_names = None
            if hasattr(self.model, 'feature_names') and self.model.feature_names:
                feature_names = self.model.feature_names
            elif hasattr(self.model, 'feature_names_in_'):
                feature_names = list(self.model.feature_names_in_)
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'feature_names_in_'):
                feature_names = list(self.model.model.feature_names_in_)
            
            # Build metadata dictionary
            metadata = {
                'model_name': model_name,
                'version': version,
                'model_type': type(self.model.model).__name__ if hasattr(self.model, 'model') else type(self.model).__name__,
                'features': feature_names if feature_names else [],
                'feature_count': len(feature_names) if feature_names else 0,
                'metrics': self.metrics,
                'cv_scores': self.cv_scores,
                'best_params': self.best_params,
                'leakage_report': self.leakage_report,
                'train_shape': self.X_train.shape if self.X_train is not None else None,
                'test_shape': self.X_test.shape if self.X_test is not None else None,
                'random_state': self.random_state,
                'timestamp': pd.Timestamp.now().isoformat(),
            }
            
            # Use dot notation (not underscore) for metadata
            metadata_path = path.parent / f"{model_name}_v{version}.metadata.json"
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            logger.info(f"✅ Metadata saved to: {metadata_path}")
            logger.info(f"   Features saved: {len(feature_names) if feature_names else 0}")
            
            return {
                "model_path": model_path,
                "metadata_path": metadata_path
            }
        
        return {"model_path": model_path}

    def get_feature_importance(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """Get feature importance with optional top-N filtering."""
        df_importance = self.model.get_feature_importance()
        
        if top_n and not df_importance.empty:
            df_importance = df_importance.head(top_n)
            logger.info(f"Top {top_n} features:")
            for _, row in df_importance.iterrows():
                logger.info(f"  {row['Feature']}: {row['Importance']:.4f}")
        
        return df_importance

    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive model report."""
        report = {
            'model_type': type(self.model.model).__name__,
            'model_params': self.model.model.get_params(),
            'best_params': self.best_params,
            'metrics': self.metrics,
            'cv_scores': self.cv_scores,
            'data_shape': {
                'train': self.X_train.shape if self.X_train is not None else None,
                'val': self.X_val.shape if self.X_val is not None else None,
                'test': self.X_test.shape if self.X_test is not None else None,
            },
            'leakage_report': self.leakage_report,
            'feature_count': self.X_train.shape[1] if self.X_train is not None else 0,
        }
        
        if self.model.is_trained:
            importance = self.model.get_feature_importance()
            if not importance.empty:
                report['top_features'] = importance.head(10).to_dict('records')
        
        return report


# ============================================================
# HYPERPARAMETER GRIDS (REDUCED FOR MEMORY)
# ============================================================

DEFAULT_RF_PARAM_GRID = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20],
    'min_samples_split': [5, 10, 20],
    'min_samples_leaf': [2, 4, 8],
    'max_features': ['sqrt', 'log2'],
}

XGBOOST_PARAM_GRID = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'min_child_weight': [1, 3],
    'gamma': [0, 0.1]
}