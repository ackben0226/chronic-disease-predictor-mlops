"""
deployment/app/dependencies.py
Dependency injection for FastAPI application.

This module handles:
1. Loading and caching the model (joblib)
2. Loading and caching metadata (JSON)
3. Providing feature names
4. Model validation

Path: deployment/app/dependencies.py
"""

import joblib
import json
import logging
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, List, Optional

# ============================================================
# IMPORTS
# ============================================================

# Use absolute import for consistency
from deployment.app.config import settings

# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    'get_model',
    'get_metadata',
    'get_feature_names',
    'validate_model',
    'get_model_or_raise',
]

# ============================================================
# LOGGER
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# MODEL LOADING
# ============================================================

@lru_cache(maxsize=1)
def get_model() -> Optional[Any]:
    """
    Load and cache the machine learning model.
    Loads ONCE at startup, cached for all subsequent requests.
    
    Returns:
        Optional[Any]: The loaded model object, or None if loading fails
    
    Raises:
        FileNotFoundError: If model file doesn't exist
        Exception: If model loading fails
    """
    
    # Convert string to Path
    model_path = Path(settings.MODEL_PATH)
    
    # Check if model exists
    if not model_path.exists():
        logger.error(f"❌ Model not found at: {model_path}")
        logger.error(f"   Current MODEL_PATH: {settings.MODEL_PATH}")
        logger.error(f"   Working directory: {Path.cwd()}")
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    try:
        logger.info(f"📂 Loading model from: {model_path}")
        model = joblib.load(model_path)
        
        # Log model info
        model_type = type(model).__name__
        if hasattr(model, 'model'):
            model_type = type(model.model).__name__
        logger.info(f"✅ Model loaded successfully")
        logger.info(f"   Type: {model_type}")
        
        # Log file size
        file_size_mb = model_path.stat().st_size / (1024 * 1024)
        logger.info(f"   Size: {file_size_mb:.2f} MB")
        
        return model
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise


# ============================================================
# METADATA LOADING
# ============================================================

@lru_cache(maxsize=1)
def get_metadata() -> Dict[str, Any]:
    """
    Load and cache model metadata.
    
    Returns:
        Dict[str, Any]: Metadata dictionary with default values if not found
    """
    
    # Convert string to Path, then get metadata path
    model_path = Path(settings.MODEL_PATH)
    metadata_path = model_path.with_suffix('.metadata.json')
    
    # Default metadata structure
    default_metadata = {
        "model_name": "Unknown",
        "version": settings.MODEL_VERSION,
        "features": [],
        "model_type": "Unknown",
        "performance": {},
        "parameters": {},
        "timestamp": None
    }
    
    if not metadata_path.exists():
        logger.warning(f"⚠️ Metadata not found at: {metadata_path}")
        logger.info(f"   Using default metadata")
        return default_metadata
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"✅ Metadata loaded successfully")
        logger.info(f"   Model: {metadata.get('model_name', 'Unknown')}")
        logger.info(f"   Version: {metadata.get('version', 'Unknown')}")
        logger.info(f"   Features: {len(metadata.get('features', []))}")
        
        return metadata
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ Failed to parse metadata JSON: {e}")
        return default_metadata
        
    except Exception as e:
        logger.error(f"❌ Failed to load metadata: {e}")
        return default_metadata


# ============================================================
# FEATURE NAMES
# ============================================================

def get_feature_names() -> List[str]:
    """
    Get the list of feature names the model expects.
    
    Returns:
        List[str]: List of feature names
    """
    metadata = get_metadata()
    features = metadata.get('features', [])
    
    if not features:
        logger.warning("⚠️ No feature names found in metadata")
    
    return features


def get_feature_count() -> int:
    """
    Get the number of features the model expects.
    
    Returns:
        int: Number of features
    """
    return len(get_feature_names())


# ============================================================
# MODEL VALIDATION
# ============================================================

def validate_model() -> bool:
    """
    Validate that the model loads correctly.
    
    Returns:
        bool: True if model loads successfully, False otherwise
    """
    try:
        model = get_model()
        if model is None:
            logger.error("❌ Model validation failed: Model is None")
            return False
        
        # Try to get feature names
        features = get_feature_names()
        logger.info(f"✅ Model validation passed")
        logger.info(f"   Features: {len(features)}")
        
        return True
        
    except FileNotFoundError as e:
        logger.error(f"❌ Model validation failed: {e}")
        return False
        
    except Exception as e:
        logger.error(f"❌ Model validation failed: {e}")
        return False


def get_model_or_raise() -> Any:
    """
    Get the model or raise a clear error.
    Use this in routes that require the model.
    
    Returns:
        Any: The loaded model
    
    Raises:
        RuntimeError: If model is not loaded
    """
    try:
        model = get_model()
        if model is None:
            raise RuntimeError("Model is not loaded")
        return model
    except FileNotFoundError as e:
        raise RuntimeError(f"Model file not found: {settings.MODEL_PATH}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}") from e


# ============================================================
# CACHE MANAGEMENT
# ============================================================

def clear_cache() -> None:
    """
    Clear all cached data (model and metadata).
    Useful for testing and reloading.
    """
    logger.info("🔄 Clearing cache...")
    get_model.cache_clear()
    get_metadata.cache_clear()
    logger.info("✅ Cache cleared")


def reload_model() -> Optional[Any]:
    """
    Reload the model (clear cache and reload).
    
    Returns:
        Optional[Any]: Reloaded model or None if failed
    """
    logger.info("🔄 Reloading model...")
    clear_cache()
    try:
        model = get_model()
        logger.info("✅ Model reloaded successfully")
        return model
    except Exception as e:
        logger.error(f"❌ Failed to reload model: {e}")
        return None


# ============================================================
# DEPENDENCY INJECTION (for FastAPI)
# ============================================================

def get_model_dependency() -> Any:
    """
    FastAPI dependency for model injection.
    Usage: 
        @app.get("/predict")
        async def predict(model: Any = Depends(get_model_dependency)):
            ...
    """
    return get_model_or_raise()


def get_metadata_dependency() -> Dict[str, Any]:
    """
    FastAPI dependency for metadata injection.
    Usage:
        @app.get("/info")
        async def info(metadata: Dict = Depends(get_metadata_dependency)):
            ...
    """
    return get_metadata()


def get_feature_names_dependency() -> List[str]:
    """
    FastAPI dependency for feature names injection.
    Usage:
        @app.post("/predict")
        async def predict(features: List[str] = Depends(get_feature_names_dependency)):
            ...
    """
    return get_feature_names()


# ============================================================
# CONVENIENCE: Get Model Info
# ============================================================

def get_model_info() -> Dict[str, Any]:
    """
    Get comprehensive model information.
    
    Returns:
        Dict[str, Any]: Model information
    """
    metadata = get_metadata()
    features = get_feature_names()
    
    return {
        "model_name": metadata.get('model_name', 'Unknown'),
        "model_version": settings.MODEL_VERSION,
        "model_type": metadata.get('model_type', 'Unknown'),
        "feature_count": len(features),
        "features": features[:20] if features else [],  # First 20
        "performance": metadata.get('performance', {}),
        "parameters": metadata.get('parameters', {}),
        "timestamp": metadata.get('timestamp', None),
        "is_loaded": validate_model()
    }


# ============================================================
# MAIN ENTRY POINT (for testing)
# ============================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("=" * 60)
    logger.info("Testing dependencies.py")
    logger.info("=" * 60)
    
    # Test model loading
    logger.info("📂 Testing model loading...")
    try:
        model = get_model()
        logger.info(f"✅ Model loaded: {type(model).__name__}")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
    
    # Test metadata loading
    logger.info("-" * 40)
    logger.info("📄 Testing metadata loading...")
    metadata = get_metadata()
    logger.info(f"✅ Metadata: {metadata.get('model_name', 'Unknown')}")
    logger.info(f"   Features: {len(metadata.get('features', []))}")
    
    # Test feature names
    logger.info("-" * 40)
    logger.info("🔍 Testing feature names...")
    features = get_feature_names()
    logger.info(f"✅ Features: {len(features)}")
    if features:
        logger.info(f"   First 5: {features[:5]}")
    
    # Test validation
    logger.info("-" * 40)
    logger.info("✅ Testing validation...")
    is_valid = validate_model()
    logger.info(f"✅ Model valid: {is_valid}")
    
    # Test model info
    logger.info("-" * 40)
    logger.info("📊 Testing model info...")
    info = get_model_info()
    logger.info(f"   Model: {info['model_name']}")
    logger.info(f"   Type: {info['model_type']}")
    logger.info(f"   Features: {info['feature_count']}")
    
    logger.info("=" * 60)
    logger.info("✅ All tests completed!")
    logger.info("=" * 60)