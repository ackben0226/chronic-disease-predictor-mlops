# 📊 Monitoring Guide

How monitoring works in the Chronic Disease Predictor.

## Overview

The system uses three layers of monitoring:

1. **Application Insights** — Azure-native monitoring
2. **Custom Drift Detection** — Feature distribution drift
3. **Health Endpoints** — Liveness and readiness probes

## Application Insights

### Setup

```bash
# Create Application Insights
az monitor app-insights component create \
  --app health-predictor-insights \
  --location eastus \
  --resource-group ml-health-predictor

# Get connection string
az monitor app-insights component show \
  --app health-predictor-insights \
  --resource-group ml-health-predictor \
  --query connectionString
