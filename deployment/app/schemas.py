"""
deployment/app/schemas.py
Pydantic models for request/response validation.

These define what data users MUST send and what they WILL receive.

Path: deployment/app/schemas.py
"""

from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ConfigDict

# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    'PredictionRequest',
    'BatchPredictionRequest',
    'PredictionResponse',
    'BatchPredictionResponse',
    'HealthResponse',
    'ErrorResponse',
    'create_error_response',
    'validate_request_data',
]


# ============================================================
# REQUEST SCHEMAS (What users send)
# ============================================================

class PredictionRequest(BaseModel):
    """
    Request schema for a single prediction.
    
    Since the model expects feature names with spaces and commas,
    we use extra="allow" to accept any fields, and the routes
    will handle the mapping to the model's expected features.
    """
    
    # ============================================================
    # CORE FEATURES (5)
    # ============================================================
    
    YearStart: int = Field(
        ...,
        ge=2000,
        le=2030,
        description="Year of the data (2000-2030)"
    )
    
    YearsSince2020: int = Field(
        ...,
        ge=0,
        le=10,
        description="Years since 2020 (0-10)"
    )
    
    ConfidenceIntervalWidth: float = Field(
        0.0,
        ge=0,
        description="Confidence interval width"
    )
    
    Latitude: float = Field(
        ...,
        ge=20,
        le=50,
        description="Latitude coordinate for US (20-50)"
    )
    
    Longitude: float = Field(
        ...,
        ge=-130,
        le=-60,
        description="Longitude coordinate for US (-130 to -60)"
    )
    
    # ============================================================
    # DATA QUALITY FLAG (1)
    # ============================================================
    
    DataValue_Missing_Flag: int = Field(
        0,
        ge=0,
        le=1,
        description="Flag indicating if DataValue was originally missing (0 or 1)"
    )
    
    # ============================================================
    # REGION FLAGS (5) - These are correct (underscores only)
    # ============================================================
    
    Region_Midwest: int = Field(
        ...,
        ge=0,
        le=1,
        description="Midwest region flag (0 or 1)"
    )
    
    Region_Northeast: int = Field(
        ...,
        ge=0,
        le=1,
        description="Northeast region flag (0 or 1)"
    )
    
    Region_South: int = Field(
        ...,
        ge=0,
        le=1,
        description="South region flag (0 or 1)"
    )
    
    Region_West: int = Field(
        ...,
        ge=0,
        le=1,
        description="West region flag (0 or 1)"
    )
    
    Region_Unknown: int = Field(
        0,
        ge=0,
        le=1,
        description="Unknown region flag (0 or 1)"
    )
    
    # ============================================================
    # DEMOGRAPHIC FLAGS
    # ============================================================
    
    # Underscore versions (for Pydantic validation)
    Demo_Overall: int = Field(
        ...,
        ge=0,
        le=1,
        description="Overall demographic flag (0 or 1)"
    )
    
    Demo_Male: int = Field(
        ...,
        ge=0,
        le=1,
        description="Male flag (0 or 1)"
    )
    
    Demo_Female: int = Field(
        ...,
        ge=0,
        le=1,
        description="Female flag (0 or 1)"
    )
    
    Demo_Age_65_plus: int = Field(
        ...,
        ge=0,
        le=1,
        description="Age 65+ flag (0 or 1)"
    )
    
    Demo_Age_18_64: int = Field(
        ...,
        ge=0,
        le=1,
        description="Age 18-64 flag (0 or 1)"
    )
    
    # Note: Fields with spaces/commas CANNOT be Pydantic field names
    # They will be accepted via extra="allow" and handled in routes
    
    # ============================================================
    # TOPIC FLAGS
    # ============================================================
    
    # Underscore versions (for Pydantic validation)
    Topic_Alcohol: int = Field(0, ge=0, le=1, description="Alcohol topic flag")
    Topic_Arthritis: int = Field(0, ge=0, le=1, description="Arthritis topic flag")
    Topic_Asthma: int = Field(0, ge=0, le=1, description="Asthma topic flag")
    Topic_Cancer: int = Field(0, ge=0, le=1, description="Cancer topic flag")
    Topic_Diabetes: int = Field(0, ge=0, le=1, description="Diabetes topic flag")
    Topic_Disability: int = Field(0, ge=0, le=1, description="Disability topic flag")
    Topic_Immunization: int = Field(0, ge=0, le=1, description="Immunization topic flag")
    Topic_Tobacco: int = Field(0, ge=0, le=1, description="Tobacco topic flag")
    
    # ============================================================
    # QUESTION FLAGS
    # ============================================================
    
    Question_Other: int = Field(0, ge=0, le=1, description="Other question flag")
    
    # ============================================================
    # CUSTOM VALIDATORS
    # ============================================================
    
    @field_validator('DataValue_Missing_Flag',
                     'Demo_Overall', 'Demo_Male', 'Demo_Female',
                     'Demo_Age_65_plus', 'Demo_Age_18_64',
                     'Region_Midwest', 'Region_Northeast', 'Region_South', 'Region_West',
                     'Region_Unknown',
                     'Topic_Alcohol', 'Topic_Arthritis', 'Topic_Asthma', 'Topic_Cancer',
                     'Topic_Diabetes', 'Topic_Disability', 'Topic_Immunization', 'Topic_Tobacco')
    @classmethod
    def validate_binary(cls, v: int) -> int:
        """Ensure binary fields are 0 or 1."""
        if v not in [0, 1]:
            raise ValueError(f'Value must be 0 or 1, got {v}')
        return v
    
    @field_validator('Latitude')
    @classmethod
    def validate_latitude(cls, v: float) -> float:
        """Ensure latitude is within US range."""
        if not (20 <= v <= 50):
            raise ValueError(f'Latitude must be between 20 and 50 for US data, got {v}')
        return v
    
    @field_validator('Longitude')
    @classmethod
    def validate_longitude(cls, v: float) -> float:
        """Ensure longitude is within US range."""
        if not (-130 <= v <= -60):
            raise ValueError(f'Longitude must be between -130 and -60 for US data, got {v}')
        return v
    
    # ============================================================
    # PYDANTIC V2 CONFIGURATION - KEY: extra="allow"
    # ============================================================
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "YearStart": 2020,
                "YearsSince2020": 0,
                "ConfidenceIntervalWidth": 0,
                "Latitude": 36.1162,
                "Longitude": -119.6816,
                "DataValue_Missing_Flag": 0,
                "Demo_Overall": 1,
                "Demo_Male": 0,
                "Demo_Female": 0,
                "Demo_Age_65_plus": 0,
                "Demo_Age_18_64": 0,
                "Region_Midwest": 0,
                "Region_Northeast": 0,
                "Region_South": 0,
                "Region_West": 1,
                "Region_Unknown": 0,
                "Topic_Alcohol": 0,
                "Topic_Arthritis": 0,
                "Topic_Asthma": 0,
                "Topic_Cancer": 0,
                "Topic_Diabetes": 1,
                "Topic_Disability": 0,
                "Topic_Immunization": 0,
                "Topic_Tobacco": 0,
                "Question_Other": 0,
                # These fields are accepted via extra="allow"
                "Demo_White, non-Hispanic": 0,
                "Demo_Black, non-Hispanic": 0,
                "Demo_Hispanic": 0,
                "Demo_Other, non-Hispanic": 0,
                "Demo_Multiracial, non-Hispanic": 0,
                "Demo_American Indian or Alaska Native": 0,
                "Demo_Asian or Pacific Islander": 0,
                "Topic_Cardiovascular Disease": 0,
                "Topic_Chronic Kidney Disease": 0,
                "Topic_Chronic Obstructive Pulmonary Disease": 0,
                "Topic_Mental Health": 0,
                "Topic_Nutrition, Physical Activity, and Weight Status": 0,
                "Topic_Older Adults": 0,
                "Topic_Oral Health": 0,
                "Topic_Overarching Conditions": 0,
                "Topic_Reproductive Health": 0,
                "Question_Arthritis among adults aged >= 18 years": 0,
                "Question_Arthritis among adults aged >= 18 years who are obese": 0,
                "Question_Arthritis among adults aged >= 18 years who have diabetes": 0,
                "Question_Arthritis among adults aged >= 18 years who have heart disease": 0,
                "Question_Asthma mortality rate": 0,
                "Question_Chronic liver disease mortality": 0,
                "Question_Dilated eye examination among adults aged >= 18 years with diagnosed diabetes": 0,
                "Question_Fair or poor health among adults aged >= 18 years with arthritis": 0,
                "Question_Mortality due to diabetes reported as any listed cause of death": 0,
                "Question_Mortality from cerebrovascular disease (stroke)": 0,
                "Question_Mortality from coronary heart disease": 0,
                "Question_Mortality from diseases of the heart": 0,
                "Question_Mortality from heart failure": 0,
                "Question_Mortality from total cardiovascular diseases": 0,
                "Question_Mortality with chronic obstructive pulmonary disease as underlying cause among adults aged >= 45 years": 0,
                "Question_Mortality with chronic obstructive pulmonary disease as underlying or contributing cause among adults aged >= 45 years": 0,
                "Question_Mortality with diabetic ketoacidosis reported as any listed cause of death": 0,
                "Question_Mortality with end-stage renal disease": 0,
                "Question_Physical inactivity among adults aged >= 18 years with arthritis": 0,
                "Question_Premature mortality among adults aged 45-64 years": 0
            }
        },
        use_enum_values=True,
        validate_assignment=True,
        extra="allow"  # ← CRITICAL: Allows fields with spaces/commas
    )


class BatchPredictionRequest(BaseModel):
    """
    Request schema for batch predictions.
    
    Allows up to 1000 predictions in a single request.
    """
    
    requests: List[PredictionRequest] = Field(
        ...,
        max_length=1000,
        description="List of prediction requests (max 1000)"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "requests": [
                    {
                        "YearStart": 2020,
                        "YearsSince2020": 0,
                        "ConfidenceIntervalWidth": 0,
                        "Latitude": 36.1162,
                        "Longitude": -119.6816,
                        "DataValue_Missing_Flag": 0,
                        "Demo_Overall": 1,
                        "Demo_Male": 0,
                        "Demo_Female": 0,
                        "Demo_Age_65_plus": 0,
                        "Demo_Age_18_64": 0,
                        "Region_Midwest": 0,
                        "Region_Northeast": 0,
                        "Region_South": 0,
                        "Region_West": 1,
                        "Region_Unknown": 0,
                        "Topic_Alcohol": 0,
                        "Topic_Arthritis": 0,
                        "Topic_Asthma": 0,
                        "Topic_Cancer": 0,
                        "Topic_Diabetes": 1,
                        "Topic_Disability": 0,
                        "Topic_Immunization": 0,
                        "Topic_Tobacco": 0,
                        "Question_Other": 0,
                        "Demo_White, non-Hispanic": 0,
                        "Topic_Cardiovascular Disease": 0,
                        "Question_Arthritis among adults aged >= 18 years": 0
                    }
                ]
            }
        }
    )


# ============================================================
# RESPONSE SCHEMAS (What users receive)
# ============================================================

class PredictionResponse(BaseModel):
    """
    Response schema for a single prediction.
    """
    
    prediction: float = Field(..., description="Predicted value")
    prediction_rounded: float = Field(..., description="Prediction rounded to 2 decimals")
    confidence_lower: Optional[float] = Field(None, description="Lower confidence bound")
    confidence_upper: Optional[float] = Field(None, description="Upper confidence bound")
    model_version: str = Field(..., description="Model version used")
    model_type: str = Field(..., description="Type of model")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    timestamp: str = Field(..., description="Prediction timestamp")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction": 15.5,
                "prediction_rounded": 15.50,
                "confidence_lower": 12.3,
                "confidence_upper": 18.7,
                "model_version": "1.0.0",
                "model_type": "RandomForestRegressor",
                "processing_time_ms": 45.23,
                "timestamp": "2024-01-15T10:30:00"
            }
        }
    )


class BatchPredictionResponse(BaseModel):
    """
    Response schema for batch predictions.
    """
    
    predictions: List[float] = Field(..., description="List of predictions")
    count: int = Field(..., description="Number of predictions")
    processing_time_ms: float = Field(..., description="Total processing time")
    timestamp: str = Field(..., description="Prediction timestamp")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predictions": [15.5, 20.3, 18.2],
                "count": 3,
                "processing_time_ms": 125.67,
                "timestamp": "2024-01-15T10:30:00"
            }
        }
    )


class HealthResponse(BaseModel):
    """
    Response schema for health check.
    """
    
    status: str = Field(..., description="Service status (healthy/unhealthy)")
    model_loaded: bool = Field(..., description="Is model loaded")
    model_version: str = Field(..., description="Model version")
    model_type: str = Field(..., description="Model type")
    features: int = Field(..., description="Number of features")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_version": "1.0.0",
                "model_type": "RandomForestRegressor",
                "features": 58,
                "uptime_seconds": 3600.5
            }
        }
    )


class ErrorResponse(BaseModel):
    """
    Response schema for errors.
    """
    
    error: str = Field(..., description="Error message")
    status_code: int = Field(..., description="HTTP status code")
    timestamp: str = Field(..., description="Error timestamp")
    details: Optional[dict] = Field(None, description="Additional error details")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "Prediction failed: Missing required fields",
                "status_code": 400,
                "timestamp": "2024-01-15T10:30:00",
                "details": {
                    "missing_fields": ["Latitude", "Longitude"]
                }
            }
        }
    )


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def create_error_response(error: str, status_code: int, details: Optional[dict] = None) -> dict:
    """
    Create a consistent error response.
    
    Args:
        error: Error message
        status_code: HTTP status code
        details: Additional error details
    
    Returns:
        dict: Error response dictionary
    """
    return {
        "error": error,
        "status_code": status_code,
        "timestamp": datetime.now().isoformat(),
        "details": details
    }


def validate_request_data(data: dict) -> dict:
    """
    Validate request data before processing.
    
    Args:
        data: Request data dictionary
    
    Returns:
        dict: Validated data
    
    Raises:
        ValueError: If validation fails
    """
    try:
        validated = PredictionRequest(**data)
        return validated.model_dump()
    except Exception as e:
        raise ValueError(f"Invalid request data: {str(e)}")


def validate_batch_request(data: dict) -> dict:
    """
    Validate batch request data before processing.
    
    Args:
        data: Batch request data dictionary
    
    Returns:
        dict: Validated data
    
    Raises:
        ValueError: If validation fails
    """
    try:
        validated = BatchPredictionRequest(**data)
        return validated.model_dump()
    except Exception as e:
        raise ValueError(f"Invalid batch request data: {str(e)}")