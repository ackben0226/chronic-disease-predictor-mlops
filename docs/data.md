# 📊 Data Documentation

CDC Chronic Disease Indicators dataset.

## Source

| Field | Value |
|-------|-------|
| **Provider** | CDC |
| **Dataset** | Chronic Disease Indicators |
| **URL** | https://www.cdc.gov/cdi/ |
| **Rows** | 403,984 |
| **Columns** | 34 |

## Schema

| Column | Type | Description |
|--------|------|-------------|
| `YearStart` | int64 | Start year |
| `YearEnd` | int64 | End year |
| `LocationAbbr` | object | State code |
| `LocationDesc` | object | State name |
| `Topic` | object | Health topic |
| `Question` | object | Metric |
| `DataValue` | object | Measurement |
| `DataValueType` | object | Metric type |
| `LowConfidenceLimit` | float64 | Lower CI |
| `HighConfidenceLimit` | float64 | Upper CI |
| `GeoLocation` | object | Lat/lon string |

## Value Ranges

| Column | Min | Max |
|--------|-----|-----|
| `YearStart` | 2001 | 2016 |
| `DataValue` | 0 | 3,967,333 |
| `LowConfidenceLimit` | 0.2 | 1,293.9 |
| `HighConfidenceLimit` | 0.42 | 2,088 |

**Note:** Confidence limits can exceed 100 because they represent rates per 100,000, not percentages.

## Data Quality

| Issue | Frequency | Handling |
|-------|-----------|----------|
| Missing DataValue | 32.3% | Smart imputation |
| Duplicates | 86 rows | Removed |
| Outliers | Small | IQR capping |

## Feature Engineering

58 features created:

- **5 core** (YearStart, Lat/Lon, etc.)
- **12 demographic** (one-hot encoded)
- **5 region** (one-hot encoded)
- **22 topic** (one-hot encoded)
- **14+ question** (one-hot encoded)

## Usage

```python
from src.ingestion.data_loading import load_data
df = load_data()
