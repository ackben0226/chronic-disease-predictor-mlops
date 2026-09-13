
# 🧠 Model Documentation

Model card for the Chronic Disease Predictor.

## Model Details

| Attribute | Value |
|-----------|-------|
| **Type** | Random Forest Regressor |
| **Version** | 1.0.0 |
| **Framework** | scikit-learn 1.3.0 |
| **Training Date** | 2026-09-12 |
| **File Size** | 214 MB |

## Intended Use

**Primary Use:** State-level predictions of chronic disease indicators.

**Intended Users:** Public health departments, policy makers.

**Out of Scope:** Individual patient diagnosis, clinical decisions.

## Performance

| Metric | Value |
|--------|-------|
| **R²** | 0.6343 |
| **RMSE** | 692.38 |
| **MAE** | 161.81 |
| **MAPE** | 189.16% |
| **Features** | 58 |

## Training Details

### Hyperparameters

```python
{
  'n_estimators': 100,
  'max_depth': 20,
  'min_samples_split': 5,
  'min_samples_leaf': 2,
  'max_features': 'sqrt',
  'random_state': 42
}
