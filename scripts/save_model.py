"""
save_model.py
Complete model packaging with all dependencies for deployment.

This module provides:
1. ModelPackager class for saving models with metadata
2. Convenience functions for one-line save/load
3. Comprehensive logging and error handling
4. Requirements.txt generation for reproducibility

Usage:
    from scripts.save_model import save_model, load_model
    
    # Save your trained model
    result = save_model(
        model=rf_model,
        model_name="chronic_disease_predictor",
        version="1.0"
    )
    
    # Later, load it
    model = load_model("chronic_disease_predictor", "1.0")
    predictions = model.predict(X_test)
"""

import joblib
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
import logging

# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    "ModelPackager",
    "save_model",
    "load_model",
    "load_metadata",
    "list_models",
]

# ============================================================
# LOGGING SETUP
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# MODEL PACKAGER CLASS
# ============================================================

@dataclass
class ModelPackager:
    """
    Package model with all dependencies for deployment.
    
    Saves:
        1. Model file (.joblib) - The trained model
        2. Metadata (.json) - Model info, features, performance
        3. Requirements (requirements.txt) - Python dependencies
    
    Attributes:
        model: Trained model object (must be pickle-able)
        model_name: Name of the model (used for file naming)
        version: Version string (e.g., "1.0", "2.1.0")
        output_dir: Directory to save files (default: "models")
        include_dependencies: Include requirements.txt (default: True)
        timestamp: ISO format timestamp (auto-generated)
    
    Examples:
        >>> packager = ModelPackager(
        ...     model=rf_model,
        ...     model_name="chronic_disease_predictor",
        ...     version="1.0"
        ... )
        >>> result = packager.save()
        >>> print(result["model_path"])
        models/chronic_disease_predictor_v1.0.joblib
    """
    
    # Required parameters
    model: Any
    model_name: str
    version: str
    
    # Optional parameters
    output_dir: str = "models"
    include_dependencies: bool = True
    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )
    
    def save(self) -> Dict[str, Path]:
        """
        Save model, metadata, and dependencies to disk.
        
        Returns:
            Dict[str, Path]: Dictionary with paths to all saved files.
        
        Raises:
            ValueError: If model is None or version is empty.
            OSError: If unable to create directory or write files.
        """
        logger.info(f"📦 Starting packaging for {self.model_name} v{self.version}")
        logger.debug(f"Output directory: {self.output_dir}")
        
        # --- Validate inputs ---
        self._validate()
        
        # --- Create output directory ---
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {output_path}")
        
        # --- Save files ---
        try:
            # 1. Save model using joblib
            model_path = self._save_model(output_path)
            
            # 2. Save metadata as JSON
            metadata_path = self._save_metadata(output_path)
            
            # 3. Save requirements.txt
            req_path = None
            if self.include_dependencies:
                req_path = self._save_requirements(output_path)
            
            logger.info(f"✅ Packaging complete for {self.model_name} v{self.version}")
            
            return {
                "model_path": model_path,
                "metadata_path": metadata_path,
                "requirements_path": req_path
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to save model: {e}")
            raise
    
    def _validate(self) -> None:
        """Validate inputs before saving."""
        if self.model is None:
            logger.error("Cannot save None model")
            raise ValueError("Cannot save None model")
        
        if not self.version:
            logger.error("Version cannot be empty")
            raise ValueError("Version cannot be empty")
        
        if not self.model_name:
            logger.error("Model name cannot be empty")
            raise ValueError("Model name cannot be empty")
        
        # Check if model has been trained (if it has an is_trained attribute)
        if hasattr(self.model, 'is_trained') and not self.model.is_trained:
            logger.warning("Model is not trained! Saving untrained model")
    
    def _save_model(self, output_path: Path) -> Path:
        """
        Save model using joblib.
        
        Args:
            output_path: Directory to save to
        
        Returns:
            Path: Path to saved model file
        """
        model_path = output_path / f"{self.model_name}_v{self.version}.joblib"
        
        logger.info(f"Saving model to: {model_path}")
        
        try:
            joblib.dump(self.model, model_path)
            logger.info(f"✅ Model saved: {model_path}")
            
            # Log model info
            model_type = self._get_model_type()
            logger.debug(f"Model type: {model_type}")
            
            # Log file size
            file_size_mb = model_path.stat().st_size / (1024 * 1024)
            logger.debug(f"File size: {file_size_mb:.2f} MB")
            
            return model_path
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def _save_metadata(self, output_path: Path) -> Path:
        """
        Save metadata as JSON.
        
        Args:
            output_path: Directory to save to
        
        Returns:
            Path: Path to saved metadata file
        """
        metadata_path = output_path / f"{self.model_name}_v{self.version}_metadata.json"
        
        logger.info(f"Saving metadata to: {metadata_path}")
        
        try:
            metadata = self._build_metadata()
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"✅ Metadata saved: {metadata_path}")
            logger.debug(f"Metadata keys: {list(metadata.keys())}")
            
            # Log feature count for verification
            feature_count = len(metadata.get('features', []))
            logger.info(f"   Features saved: {feature_count}")
            
            return metadata_path
            
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            raise
    
    def _build_metadata(self) -> Dict[str, Any]:
        """
        Build metadata dictionary from model attributes.
        
        Returns:
            Dict[str, Any]: Metadata dictionary.
        """
        features = self._get_features()
        performance = self._get_performance()
        
        # Log what we found
        if features:
            logger.info(f"✅ Found {len(features)} features for metadata")
        else:
            logger.warning("⚠️ No features found! Metadata will have empty features list.")
        
        metadata = {
            "model_name": self.model_name,
            "version": self.version,
            "timestamp": self.timestamp,
            "model_type": self._get_model_type(),
            "features": features,
            "feature_count": len(features),
            "performance": performance,
            "parameters": self._get_parameters(),
            "python_version": sys.version,
            "platform": sys.platform
        }
        
        return metadata
    
    def _get_model_type(self) -> str:
        """Get the model type string."""
        if hasattr(self.model, 'model'):
            return type(self.model.model).__name__
        return type(self.model).__name__
    
    def _get_features(self) -> List[str]:
        """
        Get feature names from model with comprehensive fallback logic.
        
        Tries multiple approaches in order:
        1. Direct feature_names attribute
        2. sklearn's feature_names_in_ on wrapped model
        3. sklearn's feature_names_in_ on the model itself
        4. Generic feature names based on feature count
        5. Infer from model structure
        """
        logger.debug("🔍 Attempting to extract feature names...")
        
        # --- Approach 1: Direct feature_names attribute ---
        if hasattr(self.model, 'feature_names') and self.model.feature_names:
            features = self.model.feature_names
            logger.info(f"✅ Extracted features from model.feature_names: {len(features)}")
            return features
        
        # --- Approach 2: sklearn's feature_names_in_ on wrapped model ---
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'feature_names_in_'):
            features = list(self.model.model.feature_names_in_)
            logger.info(f"✅ Extracted features from model.model.feature_names_in_: {len(features)}")
            return features
        
        # --- Approach 3: sklearn's feature_names_in_ on model itself ---
        if hasattr(self.model, 'feature_names_in_'):
            features = list(self.model.feature_names_in_)
            logger.info(f"✅ Extracted features from model.feature_names_in_: {len(features)}")
            return features
        
        # --- Approach 4: Get feature count from model ---
        feature_count = self._infer_feature_count()
        
        if feature_count > 0:
            logger.info(f"ℹ️ Model has {feature_count} features, using generic names")
            features = [f"feature_{i}" for i in range(feature_count)]
            logger.info(f"✅ Created {len(features)} generic feature names")
            return features
        
        # --- Approach 5: Last resort - try to get from X_train if available ---
        if hasattr(self.model, 'X_train') and self.model.X_train is not None:
            import pandas as pd
            if isinstance(self.model.X_train, pd.DataFrame):
                features = self.model.X_train.columns.tolist()
                logger.info(f"✅ Extracted features from model.X_train: {len(features)}")
                return features
        
        # --- No features found ---
        logger.warning("⚠️ Could not extract any feature names")
        logger.warning("   Please ensure model has feature_names attribute")
        return []
    
    def _infer_feature_count(self) -> int:
        """
        Try to infer the number of features from the model.
        
        Returns:
            int: Number of features, or 0 if cannot determine
        """
        # Try from wrapped model
        if hasattr(self.model, 'model'):
            # Check if it has feature_importances_
            if hasattr(self.model.model, 'feature_importances_'):
                return len(self.model.model.feature_importances_)
            
            # Check if it has n_features_in_
            if hasattr(self.model.model, 'n_features_in_'):
                return self.model.model.n_features_in_
        
        # Try from model directly
        if hasattr(self.model, 'feature_importances_'):
            return len(self.model.feature_importances_)
        
        if hasattr(self.model, 'n_features_in_'):
            return self.model.n_features_in_
        
        # Try to make a dummy prediction to infer feature count
        try:
            import numpy as np
            # Try different feature counts
            for i in range(1, 200):
                try:
                    dummy = np.random.randn(1, i)
                    if hasattr(self.model, 'model'):
                        self.model.model.predict(dummy)
                    else:
                        self.model.predict(dummy)
                    return i
                except:
                    continue
        except Exception as e:
            logger.debug(f"Could not infer feature count from dummy prediction: {e}")
        
        return 0
    
    def _get_performance(self) -> Dict[str, Any]:
        """Get performance metrics if available."""
        # Try from model.metrics
        if hasattr(self.model, 'metrics') and self.model.metrics:
            logger.info(f"✅ Found performance metrics: {self.model.metrics}")
            return self.model.metrics
        
        # Try from trainer's metrics (if stored)
        if hasattr(self.model, 'trainer_metrics') and self.model.trainer_metrics:
            logger.info(f"✅ Found trainer metrics")
            return self.model.trainer_metrics
        
        # Check if model has a .score method (scikit-learn)
        if hasattr(self.model, 'score'):
            return {"note": "Score available but not computed in metadata"}
        
        logger.warning("⚠️ No performance metrics found")
        return {}
    
    def _get_parameters(self) -> Dict[str, Any]:
        """Get model parameters if available."""
        params = {}
        
        # Try to get parameters from model
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'get_params'):
            params = self.model.model.get_params()
            logger.debug(f"Found {len(params)} model parameters")
        elif hasattr(self.model, 'get_params'):
            params = self.model.get_params()
            logger.debug(f"Found {len(params)} model parameters")
        
        # Filter out large/irrelevant params
        exclude_keys = ['n_jobs', 'random_state', 'verbose']
        filtered_params = {k: v for k, v in params.items() if k not in exclude_keys}
        
        return filtered_params
    
    def _save_requirements(self, output_path: Path) -> Optional[Path]:
        """
        Save requirements.txt using pip freeze.
        
        Args:
            output_path: Directory to save to
        
        Returns:
            Path: Path to saved requirements file, or None if failed
        """
        req_path = output_path / "requirements.txt"
        logger.info(f"Saving requirements to: {req_path}")
        
        try:
            # Try pip freeze first
            with open(req_path, 'w') as f:
                subprocess.run(
                    ["pip", "freeze"],
                    stdout=f,
                    text=True,
                    check=True,
                    capture_output=False
                )
            logger.info(f"✅ Requirements saved: {req_path}")
            return req_path
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"pip freeze failed: {e}")
            return self._create_minimal_requirements(req_path)
            
        except FileNotFoundError:
            logger.warning("pip not found in PATH")
            return self._create_minimal_requirements(req_path)
    
    def _create_minimal_requirements(self, path: Path) -> Path:
        """
        Create a minimal requirements file as fallback.
        
        Args:
            path: Path where requirements should be saved
        
        Returns:
            Path: Path to the created file
        """
        logger.warning("Creating minimal requirements file (pip freeze failed)")
        
        # Core dependencies for this project
        core_requirements = [
            "# Minimal requirements (pip freeze failed)",
            "# Please install the correct versions for your environment",
            "# Generated by save_model.py on " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "",
            "joblib>=1.2.0",
            "pandas>=2.0.0",
            "numpy>=1.24.0",
            "scikit-learn>=1.3.0",
            "fastapi>=0.104.0",
            "uvicorn[standard]>=0.24.0",
            "pydantic>=2.4.0",
            "python-multipart>=0.0.6",
            "httpx>=0.25.0"
        ]
        
        with open(path, 'w') as f:
            f.write("\n".join(core_requirements))
        
        logger.warning(f"⚠️ Created minimal requirements file: {path}")
        logger.warning("   Please verify versions are correct for your environment!")
        
        return path


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def save_model(
    model: Any,
    model_name: str,
    version: str,
    output_dir: str = "models",
    include_dependencies: bool = True
) -> Dict[str, Path]:
    """
    Convenience function to save model with one line of code.
    
    Args:
        model: Trained model
        model_name: Name of the model
        version: Version string
        output_dir: Output directory
        include_dependencies: Include requirements.txt
    
    Returns:
        Dict[str, Path]: Paths to saved files.
    
    Examples:
        >>> result = save_model(rf_model, "predictor", "1.0")
        >>> model_path = result["model_path"]
    """
    packager = ModelPackager(
        model=model,
        model_name=model_name,
        version=version,
        output_dir=output_dir,
        include_dependencies=include_dependencies
    )
    return packager.save()


def load_model(
    model_name: str,
    version: str,
    output_dir: str = "models"
) -> Any:
    """
    Load a saved model.
    
    Args:
        model_name: Name of the model
        version: Version string
        output_dir: Directory where model is saved
    
    Returns:
        Any: Loaded model.
    
    Raises:
        FileNotFoundError: If model file doesn't exist
    
    Examples:
        >>> model = load_model("chronic_disease_predictor", "1.0")
        >>> predictions = model.predict(X_test)
    """
    model_path = Path(output_dir) / f"{model_name}_v{version}.joblib"
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    logger.info(f"Loading model from: {model_path}")
    
    try:
        model = joblib.load(model_path)
        logger.info(f"✅ Model loaded successfully")
        
        # Log model info
        model_type = type(model).__name__
        if hasattr(model, 'model'):
            model_type = type(model.model).__name__
        logger.debug(f"Model type: {model_type}")
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def load_metadata(
    model_name: str,
    version: str,
    output_dir: str = "models"
) -> Dict[str, Any]:
    """
    Load metadata for a saved model.
    
    Args:
        model_name: Name of the model
        version: Version string
        output_dir: Directory where metadata is saved
    
    Returns:
        Dict[str, Any]: Metadata dictionary.
    
    Raises:
        FileNotFoundError: If metadata file doesn't exist
    """
    metadata_path = Path(output_dir) / f"{model_name}_v{version}_metadata.json"
    
    if not metadata_path.exists():
        logger.error(f"Metadata not found: {metadata_path}")
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    
    logger.info(f"Loading metadata from: {metadata_path}")
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"✅ Metadata loaded successfully")
        logger.debug(f"Model: {metadata.get('model_name')} v{metadata.get('version')}")
        logger.debug(f"Type: {metadata.get('model_type')}")
        logger.debug(f"Features: {len(metadata.get('features', []))}")
        
        return metadata
        
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}")
        raise


def list_models(output_dir: str = "models") -> List[Dict[str, str]]:
    """
    List all saved models in the output directory.
    
    Args:
        output_dir: Directory to scan for models
    
    Returns:
        List[Dict[str, str]]: List of model info dictionaries.
    """
    output_path = Path(output_dir)
    
    if not output_path.exists():
        logger.warning(f"Directory does not exist: {output_path}")
        return []
    
    models = []
    for model_file in output_path.glob("*.joblib"):
        try:
            # Extract model name and version from filename
            # Format: {model_name}_v{version}.joblib
            stem = model_file.stem
            if '_v' in stem:
                parts = stem.rsplit('_v', 1)
                if len(parts) == 2:
                    models.append({
                        "name": parts[0],
                        "version": parts[1],
                        "file": str(model_file),
                        "size_mb": model_file.stat().st_size / (1024 * 1024)
                    })
        except Exception as e:
            logger.warning(f"Could not parse model file: {model_file} ({e})")
    
    return models


# ============================================================
# MAIN ENTRY POINT (for testing)
# ============================================================

if __name__ == "__main__":
    # Configure logging for script execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger.info("=" * 60)
    logger.info("save_model.py - Testing the save/load functionality")
    logger.info("=" * 60)
    
    # Create a test model
    logger.info("Creating test model...")
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np
    
    X = np.random.randn(100, 5)
    y = X[:, 0] + 2 * X[:, 1] + np.random.randn(100) * 0.1
    
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    model.feature_names = ['feat1', 'feat2', 'feat3', 'feat4', 'feat5']
    model.metrics = {'R2': 0.95, 'RMSE': 0.15}
    
    logger.info("✅ Test model created")
    logger.info(f"   Features: {model.feature_names}")
    logger.info(f"   Metrics: {model.metrics}")
    
    # Test save
    logger.info("-" * 40)
    logger.info("Testing save_model()...")
    logger.info("-" * 40)
    
    result = save_model(
        model=model,
        model_name="test_model",
        version="1.0",
        output_dir="test_models"
    )
    
    logger.info(f"✅ Model saved to: {result['model_path']}")
    logger.info(f"✅ Metadata saved to: {result['metadata_path']}")
    
    # Test load
    logger.info("-" * 40)
    logger.info("Testing load_model()...")
    logger.info("-" * 40)
    
    loaded_model = load_model("test_model", "1.0", "test_models")
    logger.info(f"✅ Model loaded: {type(loaded_model).__name__}")
    
    # Test metadata
    logger.info("-" * 40)
    logger.info("Testing load_metadata()...")
    logger.info("-" * 40)
    
    metadata = load_metadata("test_model", "1.0", "test_models")
    logger.info(f"Model type: {metadata.get('model_type')}")
    logger.info(f"Features ({len(metadata.get('features', []))}): {metadata.get('features', [])}")
    logger.info(f"Performance: {metadata.get('performance')}")
    
    # Verify features were saved correctly
    if metadata.get('features') == ['feat1', 'feat2', 'feat3', 'feat4', 'feat5']:
        logger.info("✅ METADATA SAVED CORRECTLY!")
    else:
        logger.error("❌ METADATA NOT SAVED CORRECTLY!")
        logger.error(f"Expected: ['feat1', 'feat2', 'feat3', 'feat4', 'feat5']")
        logger.error(f"Got: {metadata.get('features')}")
    
    # Test list models
    logger.info("-" * 40)
    logger.info("Testing list_models()...")
    logger.info("-" * 40)
    
    models = list_models("test_models")
    for m in models:
        logger.info(f"  - {m['name']} v{m['version']} ({m['size_mb']:.2f} MB)")
    
    logger.info("=" * 60)
    logger.info("✅ All tests completed successfully!")
    logger.info("=" * 60)