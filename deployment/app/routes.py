"""
deployment/app/routes.py
API route definitions.

All API endpoints are defined here.

Path: deployment/app/routes.py
"""

import time
import logging
import json
from datetime import datetime
from typing import Any, Dict
from pathlib import Path

import pandas as pd

from fastapi import APIRouter, Depends, HTTPException, status

# ============================================================
# IMPORTS
# ============================================================

from .schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
)
from .dependencies import (
    get_model,
    get_metadata,
    get_feature_names,
    validate_model,
    get_model_or_raise,
)
from .config import settings

# Import security
from .security import validate_api_key, refresh_api_key, get_api_key

# Import drift detector
from monitoring.drift_detector import DriftDetector

# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    'router',
]

# ============================================================
# LOGGER
# ============================================================

logger = logging.getLogger(__name__)

# ============================================================
# ROUTER
# ============================================================

router = APIRouter()

# Track startup time
START_TIME = datetime.now()

# ============================================================
# MONITORING: Initialize Drift Detector
# ============================================================

# Initialize drift detector (loads once at startup)
drift_detector = DriftDetector(
    reference_stats_path="models/training_data_stats.json",
    threshold=2.0
)

# Track prediction count for periodic drift checks
prediction_counter = 0
DRIFT_CHECK_INTERVAL = 100  # Check drift every 100 predictions

# Create logs directory
Path("logs").mkdir(exist_ok=True)


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint for monitoring."""
    is_healthy = validate_model()
    metadata = get_metadata()
    features = get_feature_names()
    
    return HealthResponse(
        status="healthy" if is_healthy else "unhealthy",
        model_loaded=is_healthy,
        model_version=settings.MODEL_VERSION,
        model_type=metadata.get('model_type', 'Unknown'),
        features=len(features),
        uptime_seconds=(datetime.now() - START_TIME).total_seconds()
    )


# ============================================================
# MODEL INFO ENDPOINTS
# ============================================================

@router.get("/info", tags=["Model"])
async def model_info():
    """Get model information."""
    metadata = get_metadata()
    features = get_feature_names()
    
    return {
        "model": {
            "name": metadata.get('model_name', 'Unknown'),
            "version": settings.MODEL_VERSION,
            "type": metadata.get('model_type', 'Unknown'),
            "features": len(features),
            "feature_names": features[:20] if features else [],
            "performance": metadata.get('performance', {}),
            "parameters": metadata.get('parameters', {})
        },
        "api": {
            "version": settings.API_VERSION,
            "title": settings.API_TITLE,
            "docs_url": "/docs",
            "uptime_seconds": (datetime.now() - START_TIME).total_seconds()
        }
    }


@router.get("/features", tags=["Model"])
async def get_features():
    """Get list of features used by the model."""
    features = get_feature_names()
    return {
        "features": features,
        "count": len(features)
    }


# ============================================================
# PREDICTION ENDPOINT (SECURED + MONITORED)
# ============================================================

@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    request: PredictionRequest,
    api_key: str = Depends(validate_api_key),
    model: Any = Depends(get_model_or_raise),
    metadata: Dict = Depends(get_metadata)
):
    """
    Make a single prediction.
    
    Requires X-API-Key header for authentication.
    """
    start_time = time.time()
    global prediction_counter
    
    try:
        # Convert to DataFrame
        input_data = request.model_dump()
        feature_names = get_feature_names()
        
        # Build complete feature vector
        X_dict = {}
        for feature in feature_names:
            X_dict[feature] = input_data.get(feature, 0)
        
        # Log missing features
        missing_features = [f for f in feature_names if f not in input_data]
        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features, using 0")
            if len(missing_features) <= 10:
                logger.debug(f"Missing features: {missing_features}")
        
        # Create DataFrame
        df = pd.DataFrame([X_dict])
        X = df[feature_names]
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        # Try to get confidence intervals
        confidence_lower = None
        confidence_upper = None
        
        if hasattr(model, 'predict_with_confidence'):
            try:
                pred, std = model.predict_with_confidence(X)
                confidence_lower = pred[0] - 1.96 * std[0]
                confidence_upper = pred[0] + 1.96 * std[0]
            except Exception as e:
                logger.warning(f"Could not get confidence intervals: {e}")
        
        processing_time = (time.time() - start_time) * 1000
        
        # ============================================================
        # MONITORING: Log Prediction Details
        # ============================================================
        
        # 1. Log prediction metrics
        logger.info(f"📊 Prediction: {prediction:.2f}, Time: {processing_time:.2f}ms")
        
        # 2. Log confidence intervals
        if confidence_lower is not None and confidence_upper is not None:
            ci_width = confidence_upper - confidence_lower
            logger.info(f"📊 CI: [{confidence_lower:.2f}, {confidence_upper:.2f}], Width: {ci_width:.2f}")
        
        # 3. Check for anomalies
        if prediction > 1000 or prediction < 0:
            logger.warning(f"⚠️ Anomaly detected: Prediction = {prediction:.2f}")
        
        # 4. Track feature completeness
        if missing_features:
            logger.warning(f"⚠️ Missing {len(missing_features)} features: {missing_features[:5]}...")
        
        # 5. Store prediction for analysis
        prediction_record = {
            "timestamp": datetime.now().isoformat(),
            "prediction": float(prediction),
            "confidence_lower": confidence_lower,
            "confidence_upper": confidence_upper,
            "processing_time_ms": processing_time,
            "features_used": len(feature_names),
            "missing_features": len(missing_features)
        }
        
        with open("logs/predictions.log", "a") as f:
            f.write(json.dumps(prediction_record) + "\n")
        
        # ============================================================
        # MONITORING: Check Data Drift (Periodic)
        # ============================================================
        
        prediction_counter += 1
        
        if prediction_counter % DRIFT_CHECK_INTERVAL == 0:
            logger.info(f"🔍 Running drift check (prediction #{prediction_counter})")
            
            # Get a sample of recent predictions for drift detection
            drift_report = drift_detector.check_drift(df)
            
            if drift_report["drift_detected"]:
                logger.warning(
                    f"⚠️ Data drift detected! "
                    f"{drift_report['summary']['features_with_drift']} out of "
                    f"{drift_report['summary']['total_features']} features drifted"
                )
                
                # Log detailed drift info
                for feature, report in drift_report["features"].items():
                    if report.get("status") == "drift_detected":
                        logger.warning(f"   - {feature}: {report.get('message', '')}")
            else:
                logger.info(f"✅ No drift detected ({drift_report['summary']['features_stable']} features stable)")
        
        # ============================================================
        # MONITORING: Check Outliers (Every 500 predictions)
        # ============================================================
        
        if prediction_counter % 500 == 0:
            outlier_report = drift_detector.detect_outliers(df)
            if outlier_report["total_outliers"] > 0:
                logger.warning(
                    f"⚠️ Found {outlier_report['total_outliers']} outliers "
                    f"across {len(outlier_report['features'])} features"
                )
        
        # ============================================================
        # RESPONSE
        # ============================================================
        
        return PredictionResponse(
            prediction=float(prediction),
            prediction_rounded=round(float(prediction), 2),
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
            model_version=settings.MODEL_VERSION,
            model_type=metadata.get('model_type', 'Unknown'),
            processing_time_ms=round(processing_time, 2),
            timestamp=datetime.now().isoformat()
        )
    
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {str(e)}"
        )
    
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


# ============================================================
# BATCH PREDICTION ENDPOINT (SECURED + MONITORED)
# ============================================================

@router.post("/predict_batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(
    request: BatchPredictionRequest,
    api_key: str = Depends(validate_api_key),
    model: Any = Depends(get_model_or_raise),
    metadata: Dict = Depends(get_metadata)
):
    """
    Make batch predictions.
    
    Requires X-API-Key header for authentication.
    """
    start_time = time.time()
    
    try:
        # Convert to list of dicts
        data = [req.model_dump() for req in request.requests]
        feature_names = get_feature_names()
        
        # Build complete feature vectors for all requests
        feature_data = []
        for row in data:
            X_dict = {}
            for feature in feature_names:
                X_dict[feature] = row.get(feature, 0)
            feature_data.append(X_dict)
        
        # Create DataFrame
        df = pd.DataFrame(feature_data)
        X = df[feature_names]
        
        # Make predictions
        predictions = model.predict(X)
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(f"Batch: {len(predictions)} predictions in {processing_time:.2f}ms")
        
        return BatchPredictionResponse(
            predictions=predictions.tolist(),
            count=len(predictions),
            processing_time_ms=round(processing_time, 2),
            timestamp=datetime.now().isoformat()
        )
    
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {str(e)}"
        )
    
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


# ============================================================
# MONITORING ENDPOINTS
# ============================================================

@router.get("/monitoring/health", tags=["Monitoring"])
async def monitoring_health():
    """
    Detailed health check for monitoring.
    Provides comprehensive system health status.
    """
    try:
        is_healthy = validate_model()
        metadata = get_metadata()
        features = get_feature_names()
        
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "uptime_seconds": (datetime.now() - START_TIME).total_seconds(),
            "model_loaded": is_healthy,
            "features_count": len(features),
            "model_type": metadata.get('model_type', 'Unknown'),
            "model_version": settings.MODEL_VERSION,
            "prediction_count": prediction_counter,
            "drift_check_interval": DRIFT_CHECK_INTERVAL,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Monitoring health check failed: {e}")
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.get("/monitoring/metrics", tags=["Monitoring"])
async def get_metrics():
    """
    Get monitoring metrics.
    Returns performance and usage metrics.
    """
    try:
        metadata = get_metadata()
        features = get_feature_names()
        
        # Get drift detector summary
        drift_summary = drift_detector.get_summary()
        
        # Get prediction logs stats
        log_file = Path("logs/predictions.log")
        prediction_stats = {
            "total_predictions": prediction_counter,
            "log_file_exists": log_file.exists(),
            "log_file_size_mb": round(log_file.stat().st_size / (1024 * 1024), 2) if log_file.exists() else 0
        }
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "model_performance": metadata.get('performance', {}),
            "system_health": {
                "status": "healthy",
                "uptime_seconds": (datetime.now() - START_TIME).total_seconds(),
                "model_loaded": validate_model(),
                "features_count": len(features)
            },
            "drift_monitoring": {
                "features_monitored": drift_summary.get('features_monitored', 0),
                "threshold": drift_summary.get('threshold', 2.0),
                "reference_created_at": drift_summary.get('reference_created_at', 'unknown')
            },
            "prediction_stats": prediction_stats
        }
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.post("/monitoring/drift-check", tags=["Monitoring"])
async def check_drift():
    """
    Manual drift check endpoint.
    Returns current drift report for all monitored features.
    """
    try:
        # Get drift report
        drift_report = drift_detector.check_drift(pd.DataFrame({}))  # Empty check
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "drift_detected": drift_report.get("drift_detected", False),
            "summary": drift_report.get("summary", {}),
            "features": drift_report.get("features", {}),
            "message": "Drift check completed. To run a full check, send a sample of current data."
        }
    except Exception as e:
        logger.error(f"Drift check failed: {e}")
        return {
            "status": "error",
            "message": f"Drift check failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }


@router.get("/monitoring/drift-report", tags=["Monitoring"])
async def get_drift_report():
    """
    Get the current drift report summary.
    """
    try:
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "drift_detector_summary": drift_detector.get_summary(),
            "prediction_count": prediction_counter,
            "drift_check_interval": DRIFT_CHECK_INTERVAL,
            "features_monitored": len(drift_detector.reference_stats.get("features", {}))
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@router.post("/monitoring/save-reference-stats", tags=["Monitoring"])
async def save_reference_stats():
    """
    Save reference statistics from training data.
    This endpoint would be used during model training.
    """
    try:
        # This would normally use training data
        # For now, we'll just return a message
        return {
            "status": "success",
            "message": "Reference stats saved. Train your model first to generate stats.",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to save reference stats: {str(e)}"
        }


# ============================================================
# ADMIN ENDPOINTS
# ============================================================

@router.post("/admin/refresh-key", tags=["Admin"])
async def refresh_key():
    """Refresh API key from Key Vault."""
    try:
        new_key = refresh_api_key()
        return {
            "status": "success",
            "message": "API key refreshed successfully",
            "key_preview": new_key[:8] + "..."
        }
    except Exception as e:
        logger.error(f"Failed to refresh key: {e}")
        return {
            "status": "error",
            "message": f"Failed to refresh key: {str(e)}"
        }


@router.get("/admin/get-key", tags=["Admin"])
async def get_current_key():
    """Get the current API key (partial)."""
    try:
        full_key = get_api_key()
        return {
            "status": "success",
            "key_preview": full_key[:8] + "..."
        }
    except Exception as e:
        logger.error(f"Failed to get key: {e}")
        return {
            "status": "error",
            "message": f"Failed to get key: {str(e)}"
        }


# ============================================================
# ERROR HANDLERS
# ============================================================

@router.get("/health/liveness", tags=["Health"])
async def liveness():
    """Liveness probe for Kubernetes."""
    return {"status": "alive"}


@router.get("/health/readiness", tags=["Health"])
async def readiness():
    """Readiness probe for Kubernetes."""
    is_healthy = validate_model()
    return {
        "status": "ready" if is_healthy else "not_ready",
        "model_loaded": is_healthy
    }


# ============================================================
# ADDITIONAL UTILITY ENDPOINTS
# ============================================================

@router.get("/model/validate", tags=["Model"])
async def validate_model_endpoint():
    """Validate that the model is working correctly."""
    try:
        model = get_model_or_raise()
        features = get_feature_names()
        
        dummy_data = {feature: 0 for feature in features}
        dummy_data.update({
            "YearStart": 2020,
            "Latitude": 36.0,
            "Longitude": -115.0,
            "Demo_Overall": 1,
            "Region_West": 1
        })
        
        df = pd.DataFrame([dummy_data])
        X = df[features]
        prediction = model.predict(X)[0]
        
        return {
            "status": "success",
            "message": "Model validation passed",
            "test_prediction": float(prediction),
            "features_tested": len(features)
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Model validation failed: {str(e)}"
        }


@router.post("/predict/health", tags=["Prediction"])
async def predict_health_check():
    """Health check that validates prediction path."""
    try:
        feature_names = get_feature_names()
        
        if not feature_names:
            return {
                "status": "error",
                "message": "No feature names available"
            }
        
        dummy_data = {}
        for feature in feature_names:
            dummy_data[feature] = 0
        
        dummy_data.update({
            "YearStart": 2020,
            "Latitude": 36.0,
            "Longitude": -115.0,
            "Demo_Overall": 1,
            "Region_West": 1,
            "Topic_Diabetes": 1,
            "Question_Prevalence": 1
        })
        
        request = PredictionRequest(**dummy_data)
        
        return {
            "status": "success",
            "message": "Prediction path is healthy",
            "features_used": len(feature_names)
        }
    
    except Exception as e:
        logger.error(f"Prediction health check failed: {e}")
        return {
            "status": "error",
            "message": f"Prediction path failed: {str(e)}"
        }