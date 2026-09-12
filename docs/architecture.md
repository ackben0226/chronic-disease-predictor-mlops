# 🏗️ Architecture

System architecture documentation for the Chronic Disease Predictor.

## Overview

The system is a complete MLOps pipeline that spans data ingestion, model training, deployment, and monitoring. Each component is containerized and deployed on Azure.

## High-Level Design
## High-Level Architecture

```mermaid
flowchart LR

    DS[Data Sources]
    DP[Data Pipeline]
    FS[Feature Store]
    MT[Model Training]
    MR[Model Registry]

    C[Clients]
    GW[API Gateway]
    API[ML API<br/>FastAPI]
    R[Response]

    KV[Key Vault<br/>Secrets]
    CICD[CI/CD<br/>GitHub]
    MON[Monitoring<br/>Application Insights]

    DS --> DP
    DP --> FS
    FS --> MT
    MT --> MR

    MR --> API

    C --> GW
    GW --> API
    API --> R

    GW --> KV

    CICD -.-> MT
    CICD -.-> API

    API --> MON
```


## Components

### 1. Data Pipeline (`src/`)

Handles all data operations from raw CSV to ML-ready features.
