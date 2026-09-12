"""
deployment/app/config.py
Configuration management for the FastAPI application.

This file contains ALL configuration settings.
Environment variables can override defaults.

Path: deployment/app/config.py
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Union
from dataclasses import dataclass, field


# ============================================================
# LOGGER SETUP
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _parse_origins(origins: str) -> List[str]:
    """
    Parse comma-separated origins string into a list.
    
    Args:
        origins: Comma-separated string, or "*" for all
    
    Returns:
        List[str]: List of allowed origins
    
    Examples:
        >>> _parse_origins("*")
        ['*']
        >>> _parse_origins("http://localhost:3000,https://example.com")
        ['http://localhost:3000', 'https://example.com']
    """
    if not origins or origins == "*":
        return ["*"]
    return [o.strip() for o in origins.split(",") if o.strip()]


def _parse_bool(value: Optional[str]) -> bool:
    """Parse boolean from environment variable."""
    if value is None:
        return False
    return value.lower() in ("true", "1", "yes", "on")


def _parse_int(value: Optional[str], default: int) -> int:
    """Parse integer from environment variable."""
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


# ============================================================
# VALID LOG LEVELS
# ============================================================

VALID_LOG_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']


# ============================================================
# SETTINGS CLASS
# ============================================================

@dataclass
class Settings:
    """
    Application settings.
    
    All settings can be overridden by environment variables.
    Using @dataclass for cleaner configuration management.
    
    Environment Variables:
        MODEL_PATH: Path to model file
        MODEL_VERSION: Model version string
        HOST: Host to bind to
        PORT: Port to bind to
        RELOAD: Enable auto-reload (true/false)
        WORKERS: Number of worker processes
        ALLOWED_ORIGINS: Comma-separated CORS origins
        LOG_LEVEL: Logging level
    """
    
    # ===== API CONFIGURATION =====
    API_TITLE: str = "Health Predictor API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = """
    Chronic Disease Indicator Prediction API.
    
    This API serves predictions from the trained Random Forest model
    for CDC Chronic Disease Indicators data.
    """
    
    # ===== MODEL CONFIGURATION =====
    MODEL_PATH: str = field(
        default_factory=lambda: os.getenv(
            "MODEL_PATH",
            "models/chronic_disease_predictor_v1.0.joblib"
        )
    )
    
    MODEL_VERSION: str = field(
        default_factory=lambda: os.getenv("MODEL_VERSION", "1.0.0")
    )
    
    # ===== SERVER CONFIGURATION =====
    HOST: str = field(
        default_factory=lambda: os.getenv("HOST", "0.0.0.0")
    )
    
    PORT: int = field(
        default_factory=lambda: _parse_int(os.getenv("PORT"), 8000)
    )
    
    RELOAD: bool = field(
        default_factory=lambda: _parse_bool(os.getenv("RELOAD"))
    )
    
    WORKERS: int = field(
        default_factory=lambda: _parse_int(os.getenv("WORKERS"), 4)
    )
    
    # ===== CORS CONFIGURATION =====
    ALLOWED_ORIGINS: List[str] = field(
        default_factory=lambda: _parse_origins(
            os.getenv("ALLOWED_ORIGINS", "*")
        )
    )
    
    # ===== LOGGING =====
    LOG_LEVEL: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )
    
    # ===== SECURITY =====
    API_KEY: Optional[str] = field(
        default_factory=lambda: os.getenv("API_KEY", None)
    )
    
    # ============================================================
    # VALIDATION
    # ============================================================
    
    def validate(self) -> bool:
        """
        Validate that all configuration settings are correct.
        
        Checks:
        - Model file exists
        - Port is valid (1-65535)
        - Workers is positive
        - Log level is valid
        
        Returns:
            bool: True if validation passes, False otherwise
        """
        logger.info("🔍 Validating configuration settings...")
        all_valid = True
        
        # 1. Validate model path
        model_path = Path(self.MODEL_PATH)
        if not model_path.exists():
            logger.error(f"❌ Model not found at: {model_path}")
            logger.error(f"   Please ensure the model file exists.")
            logger.error(f"   Current MODEL_PATH: {self.MODEL_PATH}")
            logger.error(f"   Working directory: {Path.cwd()}")
            all_valid = False
        else:
            logger.info(f"✅ Model found at: {model_path}")
            file_size_mb = model_path.stat().st_size / (1024 * 1024)
            logger.info(f"   File size: {file_size_mb:.2f} MB")
        
        # 2. Validate port
        if not (1 <= self.PORT <= 65535):
            logger.error(f"❌ Invalid PORT: {self.PORT} (must be 1-65535)")
            all_valid = False
        else:
            logger.info(f"✅ Port: {self.PORT}")
        
        # 3. Validate workers
        if self.WORKERS < 1:
            logger.error(f"❌ Invalid WORKERS: {self.WORKERS} (must be >= 1)")
            all_valid = False
        else:
            logger.info(f"✅ Workers: {self.WORKERS}")
        
        # 4. Validate log level
        if self.LOG_LEVEL.upper() not in VALID_LOG_LEVELS:
            logger.warning(
                f"⚠️ Unknown LOG_LEVEL: {self.LOG_LEVEL}, using INFO"
            )
        else:
            logger.info(f"✅ Log Level: {self.LOG_LEVEL}")
        
        return all_valid
    
    def log_config(self) -> None:
        """
        Log all configuration settings (without sensitive data).
        Useful for debugging and startup.
        """
        logger.info("=" * 60)
        logger.info("📋 Configuration Settings")
        logger.info("=" * 60)
        
        # API settings
        logger.info(f"API Title: {self.API_TITLE}")
        logger.info(f"API Version: {self.API_VERSION}")
        
        # Model settings
        logger.info(f"Model Path: {self.MODEL_PATH}")
        logger.info(f"Model Version: {self.MODEL_VERSION}")
        
        # Server settings
        logger.info(f"Host: {self.HOST}")
        logger.info(f"Port: {self.PORT}")
        logger.info(f"Workers: {self.WORKERS}")
        logger.info(f"Reload: {self.RELOAD}")
        
        # Logging
        logger.info(f"Log Level: {self.LOG_LEVEL}")
        
        # CORS
        origins = self.ALLOWED_ORIGINS
        if origins == ["*"]:
            logger.info("CORS Origins: * (all origins)")
        else:
            logger.info(f"CORS Origins: {origins}")
        
        logger.info("=" * 60)


# ============================================================
# CREATE SETTINGS INSTANCE
# ============================================================

settings = Settings()


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def get_model_path() -> Path:
    """Get the model path as a Path object."""
    return Path(settings.MODEL_PATH)


def get_model_version() -> str:
    """Get the model version."""
    return settings.MODEL_VERSION


def is_development() -> bool:
    """Check if running in development mode."""
    return settings.RELOAD is True


def get_api_info() -> dict:
    """Get API information as a dictionary."""
    return {
        "title": settings.API_TITLE,
        "version": settings.API_VERSION,
        "description": settings.API_DESCRIPTION,
        "docs_url": "/docs"
    }


def reload_settings() -> Settings:
    """
    Reload settings from environment variables.
    Useful for testing or dynamic configuration changes.
    """
    logger.info("🔄 Reloading settings...")
    
    # Reload settings
    global settings
    settings = Settings()
    
    # Log new configuration
    settings.log_config()
    
    # Validate
    is_valid = settings.validate()
    
    if is_valid:
        logger.info("✅ Settings reloaded successfully")
    else:
        logger.error("❌ Settings reload failed")
    
    return settings


def validate_config_on_startup() -> bool:
    """
    Validate configuration on startup.
    Called from the main application.
    
    Returns:
        bool: True if valid, False otherwise
    """
    logger.info("=" * 60)
    logger.info("🔍 Starting configuration validation...")
    logger.info("=" * 60)
    
    settings.log_config()
    is_valid = settings.validate()
    
    if not is_valid:
        logger.error("=" * 60)
        logger.error("❌ Configuration validation FAILED!")
        logger.error("   The API may not work correctly.")
        logger.error("   Please check your configuration.")
        logger.error("=" * 60)
        return False
    else:
        logger.info("=" * 60)
        logger.info("✅ Configuration validation PASSED!")
        logger.info("=" * 60)
        return True