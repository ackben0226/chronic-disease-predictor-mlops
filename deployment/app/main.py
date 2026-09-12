"""
deployment/app/main.py
FastAPI application entry point.

This is where the FastAPI application is created, configured,
and prepared to serve your ML model.

Path: deployment/app/main.py
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ============================================================
# IMPORTS FROM SAME PACKAGE
# ============================================================

from .config import settings
from .dependencies import get_model, get_metadata
from .routes import router

# ============================================================
# SCHEMAS (for type hints and responses)
# ============================================================

from .schemas import (
    HealthResponse,
    ErrorResponse,
)


# ============================================================
# LOGGING SETUP
# ============================================================

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


# ============================================================
# LIFECYCLE MANAGEMENT (Startup & Shutdown)
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handle startup and shutdown events.

    What happens at startup:
    - Model is loaded
    - Metadata is loaded
    - Dependencies are initialized

    What happens at shutdown:
    - Clean up resources (if needed)
    """
    # ===== STARTUP =====
    logger.info("=" * 60)
    logger.info("🚀 Starting Health Predictor API")
    logger.info("=" * 60)

    try:
        # Load model at startup
        model = get_model()
        metadata = get_metadata()

        logger.info(f"✅ Model loaded successfully")
        logger.info(f"   Model: {metadata.get('model_name', 'Unknown')}")
        logger.info(f"   Version: {settings.MODEL_VERSION}")
        logger.info(f"   Features: {len(metadata.get('features', []))}")
        logger.info(f"   Type: {metadata.get('model_type', 'Unknown')}")

    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise

    # The application runs here
    yield

    # ===== SHUTDOWN =====
    logger.info("=" * 60)
    logger.info("🛑 Shutting down Health Predictor API")
    logger.info("=" * 60)


# ============================================================
# CREATE FASTAPI APPLICATION
# ============================================================

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    This factory pattern allows:
    - Easy testing (create test app with different configs)
    - Multiple app instances if needed
    - Clean separation of concerns

    Returns:
        FastAPI: Configured FastAPI application
    """

    # Create the app
    app = FastAPI(
        title=settings.API_TITLE,
        version=settings.API_VERSION,
        description=settings.API_DESCRIPTION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )

    # ===== MIDDLEWARE =====

    # CORS Middleware - Allows frontend/other services to call your API
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ===== ROUTES =====

    # Include all routes from routes.py
    app.include_router(router)

    # ===== ROOT ENDPOINT =====

    @app.get("/")
    async def root():
        """
        Root endpoint - shows API information.
        """
        return {
            "message": "Health Predictor API",
            "version": settings.API_VERSION,
            "docs": "/docs",
            "status": "healthy"
        }

    # ===== HEALTH CHECK =====

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """
        Health check for monitoring.
        Uses the HealthResponse schema.
        """
        try:
            # Verify model is loaded
            model = get_model()
            metadata = get_metadata()

            return JSONResponse(
                status_code=200,
                content={
                    "status": "healthy",
                    "model_loaded": model is not None,
                    "model_version": settings.MODEL_VERSION,
                    "model_type": metadata.get('model_type', 'Unknown'),
                    "features": len(metadata.get('features', [])),
                    "uptime_seconds": 0.0
                }
            )

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "model_loaded": False,
                    "model_version": settings.MODEL_VERSION,
                    "model_type": "Unknown",
                    "features": 0,
                    "uptime_seconds": 0.0
                }
            )

    return app


# ============================================================
# APP INSTANCE
# ============================================================

# Create the app instance
app = create_app()


# ============================================================
# MAIN ENTRY POINT (for running directly)
# ============================================================

if __name__ == "__main__":
    """
    Run the app directly (for development).

    Example:
        python -m deployment.app.main
    """
    import uvicorn

    uvicorn.run(
        "deployment.app.main:app",  # Where the app is
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        workers=settings.WORKERS,
        log_level=settings.LOG_LEVEL.lower()
    )