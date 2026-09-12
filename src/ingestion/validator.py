"""
Data validation module.
Ensures the raw data meets basic quality checks before processing.
"""

import logging
from typing import Optional, Dict, Tuple, List

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# PART 1: RAW DATA VALIDATION
# ============================================================

def validate_schema(df: pd.DataFrame, required_columns: list) -> None:
    """
    Check if all required columns exist in the DataFrame.

    Args:
        df: Input DataFrame.
        required_columns: List of column names that MUST be present.

    Raises:
        ValueError: If any required column is missing.
    """
    missing_cols = [col for col in required_columns if col not in df.columns]

    if missing_cols:
        logger.error(f"Schema validation failed! Missing columns: {missing_cols}")
        logger.info(f"Available columns: {df.columns.tolist()}")
        raise ValueError(f"Missing required columns: {missing_cols}")

    logger.info(f"Schema validation passed. All {len(required_columns)} present.")


def validate_not_empty(df: pd.DataFrame) -> None:
    """
    Check if the DataFrame has at least one row.

    Raises:
        ValueError: If DataFrame is empty.
    """
    if df.empty:
        logger.error("Data validation failed: DataFrame is completely empty.")
        raise ValueError("Loaded CSV is empty. Check the raw file.")

    logger.info(f"Data size validation passed. Rows: {df.shape[0]:,}")


def get_bounds_for_datavaluetype(vtype: str, col: str) -> Optional[Tuple[float, float]]:
    """
    Get appropriate bounds for a specific DataValueType and column.
    
    Args:
        vtype: DataValueType string (e.g., 'Crude Prevalence', 'Age-adjusted Rate')
        col: Column name ('DataValue', 'LowConfidenceLimit', 'HighConfidenceLimit')
    
    Returns:
        Tuple of (min_val, max_val) or None if unknown type
    """
    vtype_lower = str(vtype).lower()
    
    # Determine metric type
    is_percentage = any(keyword in vtype_lower for keyword in ['prevalence', 'percent', 'percentage'])
    is_rate = any(keyword in vtype_lower for keyword in ['rate']) and not is_percentage
    is_count = any(keyword in vtype_lower for keyword in ['number', 'count'])
    is_yes_no = any(keyword in vtype_lower for keyword in ['yes/no'])
    is_mean = any(keyword in vtype_lower for keyword in ['mean', 'average'])
    is_median = 'median' in vtype_lower
    is_dollars = 'dollars' in vtype_lower
    
    # For confidence limits, use wider bounds
    if "confidence" in col.lower():
        if is_percentage:
            return (0, 100)
        elif is_rate:
            return (0, 5000)  # Rates per 100,000 can be very high
        elif is_count:
            return (0, 10_000_000)  # Counts can be large
        elif is_yes_no:
            return (0, 1)
        elif is_dollars:
            return (0, 1_000_000)
        else:
            return (0, 10000)  # Broad fallback for means/medians
    
    # For DataValue itself
    if col == "DataValue" or col == "DataValueAlt":
        if is_percentage:
            return (0, 100)
        elif is_rate:
            return (0, 5000)
        elif is_count:
            return (0, 10_000_000)
        elif is_yes_no:
            return (0, 1)
        elif is_mean or is_median:
            return (0, 10000)
        elif is_dollars:
            return (0, 1_000_000)
        else:
            return (0, 10000)  # Broad fallback
    
    return None


def validate_value_ranges(df: pd.DataFrame, numeric_checks: Dict[str, Tuple[float, float]]) -> None:
    """
    Check if numeric columns fall within expected min/max bounds.
    Validates per DataValueType for context-aware bounds.

    Args:
        df: Input DataFrame.
        numeric_checks: Dictionary where key = column name,
                        value = tuple of (min_val, max_val).

    Raises:
        ValueError: If values fall outside the allowed range.
    """
    for col, (min_val, max_val) in numeric_checks.items():
        if col not in df.columns:
            logger.warning(f"Skipping range check for '{col}' (column missing).")
            continue

        # Try to convert to numeric, coerce errors to NaN
        series = pd.to_numeric(df[col], errors="coerce")
        
        # If we have DataValueType and this is a metric column, validate per type
        if "DataValueType" in df.columns and col in ["DataValue", "DataValueAlt", "LowConfidenceLimit", "HighConfidenceLimit"]:
            # Group by DataValueType and validate each separately
            all_passed = True
            value_types = df["DataValueType"].dropna().unique()
            
            # Track results for logging
            type_results = []
            
            for vtype in value_types:
                mask = df["DataValueType"] == vtype
                type_series = series[mask].dropna()
                
                if type_series.empty:
                    continue
                
                # Get bounds for this specific DataValueType
                type_bounds = get_bounds_for_datavaluetype(vtype, col)
                if type_bounds is None:
                    # Use the global bounds as fallback
                    type_bounds = (min_val, max_val)
                
                type_min, type_max = type_bounds
                out_of_bounds = (type_series < type_min) | (type_series > type_max)
                invalid_count = out_of_bounds.sum()
                
                if invalid_count > 0:
                    invalid_pct = invalid_count / len(type_series) * 100
                    sample_values = type_series[out_of_bounds].head(5).tolist()
                    extreme_min = type_series[out_of_bounds].min() if invalid_count > 0 else None
                    extreme_max = type_series[out_of_bounds].max() if invalid_count > 0 else None
                    
                    # If more than 5% are outside, flag it as error
                    if invalid_pct > 5:
                        logger.error(
                            f"Column '{col}' for DataValueType '{vtype}' has {invalid_count:,} values "
                            f"({invalid_pct:.1f}%) outside [{type_min}, {type_max}]. "
                            f"Range: {extreme_min:.2f} - {extreme_max:.2f}. Samples: {sample_values}"
                        )
                        all_passed = False
                    else:
                        logger.warning(
                            f"Column '{col}' for DataValueType '{vtype}' has {invalid_count:,} values "
                            f"({invalid_pct:.1f}%) outside [{type_min}, {type_max}]. "
                            f"Range: {extreme_min:.2f} - {extreme_max:.2f}. Samples: {sample_values}"
                        )
                    
                    type_results.append({
                        'type': vtype,
                        'total': len(type_series),
                        'invalid': invalid_count,
                        'pct': invalid_pct,
                        'passed': invalid_pct <= 5
                    })
                else:
                    type_results.append({
                        'type': vtype,
                        'total': len(type_series),
                        'invalid': 0,
                        'pct': 0,
                        'passed': True
                    })
            
            # Log summary
            passed_count = sum(1 for r in type_results if r['passed'])
            logger.info(
                f"Column '{col}': {passed_count}/{len(type_results)} DataValueTypes passed validation"
            )
            
            if not all_passed:
                failed_types = [r['type'] for r in type_results if not r['passed']]
                raise ValueError(
                    f"Column '{col}' has values outside expected ranges for: {failed_types}. "
                    f"Check logs for details."
                )
            else:
                logger.info(f"Column '{col}' validation passed per DataValueType.")
        
        else:
            # Standard validation for non-DataValue columns
            series_clean = series.dropna()
            if series_clean.empty:
                logger.warning(f"No numeric values found in '{col}'. Skipping range check.")
                continue

            out_of_bounds = (series_clean < min_val) | (series_clean > max_val)
            invalid_count = out_of_bounds.sum()

            if invalid_count > 0:
                invalid_pct = invalid_count / len(series_clean) * 100
                sample_values = series_clean[out_of_bounds].head(5).tolist()
                
                if invalid_pct > 5:
                    raise ValueError(
                        f"Column '{col}' has {invalid_count:,} values ({invalid_pct:.1f}%) "
                        f"outside [{min_val}, {max_val}]. Samples: {sample_values}"
                    )
                else:
                    logger.warning(
                        f"Column '{col}' has {invalid_count:,} values ({invalid_pct:.1f}%) "
                        f"outside [{min_val}, {max_val}]. Samples: {sample_values}"
                    )

    logger.info("Value range validation passed.")


def validate_basic_quality(df: pd.DataFrame) -> None:
    """
    Quick sanity checks for data quality.
    Handles CDC dataset specifics: YearStart, DataValue as object, etc.
    """
    # ============================================================
    # 1. YEAR VALIDATION - Handle YearStart or Year
    # ============================================================
    year_col = None
    if "Year" in df.columns:
        year_col = "Year"
    elif "YearStart" in df.columns:
        year_col = "YearStart"
        
    if year_col:
        year_series = pd.to_numeric(df[year_col], errors="coerce")
        valid_years = year_series.dropna()
        
        if len(valid_years) > 0:
            median_year = valid_years.median()
            min_year = valid_years.min()
            max_year = valid_years.max()
            
            logger.info(f"{year_col} range: {min_year:.0f} - {max_year:.0f} (median: {median_year:.0f})")
            
            if median_year < 1990:
                logger.warning(
                    f"Detected unusually low median {year_col}: {median_year:.0f}. "
                    f"Check if data contains historical records or parsing errors."
                )
            elif median_year > 2030:
                logger.warning(
                    f"Detected unusually high median {year_col}: {median_year:.0f}. "
                    f"Data might contain future estimates or errors."
                )
        else:
            logger.warning(f"Column '{year_col}' exists but all values are NaN or invalid.")
    else:
        logger.warning("Neither 'Year' nor 'YearStart' column found. Skipping year validation.")

    # ============================================================
    # 2. YEAREND VALIDATION (if available)
    # ============================================================
    if "YearEnd" in df.columns:
        year_end_series = pd.to_numeric(df["YearEnd"], errors="coerce")
        valid_end_years = year_end_series.dropna()
        
        if len(valid_end_years) > 0:
            median_end = valid_end_years.median()
            min_end = valid_end_years.min()
            max_end = valid_end_years.max()
            
            logger.info(f"YearEnd range: {min_end:.0f} - {max_end:.0f} (median: {median_end:.0f})")
            
            if median_end < 2000 or median_end > 2030:
                logger.warning(
                    f"Unusual YearEnd median: {median_end:.0f}. "
                    f"Data might include future projections or historical records."
                )

    # ============================================================
    # 3. DATA VALUE VALIDATION (Critical for CDC data)
    # ============================================================
    value_col = None
    if "DataValue" in df.columns:
        value_col = "DataValue"
    elif "DataValueAlt" in df.columns:
        value_col = "DataValueAlt"
    
    if value_col:
        # CDC stores DataValue as object with 'Missing', 'Suppressed', etc.
        numeric_values = pd.to_numeric(df[value_col], errors="coerce")
        valid_values = numeric_values.dropna()
        
        if len(valid_values) > 0:
            valid_pct = len(valid_values) / len(df) * 100
            logger.info(
                f"DataValue valid numeric rows: {len(valid_values):,} / {len(df):,} "
                f"({valid_pct:.1f}%)"
            )
            
            # Check for extreme values
            value_min = valid_values.min()
            value_max = valid_values.max()
            value_median = valid_values.median()
            
            logger.info(f"DataValue stats: min={value_min:.2f}, median={value_median:.2f}, max={value_max:.2f}")
            
            if value_min < 0:
                logger.warning(
                    f"Negative {value_col} detected: min = {value_min:.2f}. "
                    f"Check if data contains percentages below 0 or errors."
                )
            
            # Check if this is percentage data
            is_percentage = False
            if "DataValueType" in df.columns:
                sample_types = df["DataValueType"].dropna().unique()
                is_percentage = any(
                    'Prevalence' in str(t) or 
                    'Percent' in str(t) or 
                    'Percentage' in str(t)
                    for t in sample_types
                )
            
            if is_percentage and value_max > 100:
                logger.warning(
                    f"{value_col} > 100 for percentage data: max = {value_max:.2f}. "
                    f"Check data type or units."
                )
            
            # Log missing value patterns
            missing_pct = (len(df) - len(valid_values)) / len(df) * 100
            if missing_pct > 20:
                # Check what the non-numeric values are
                non_numeric = df[~numeric_values.notna()][value_col].value_counts().head(10)
                logger.warning(
                    f"High proportion of non-numeric DataValue: {missing_pct:.1f}%. "
                    f"Sample: {non_numeric.to_dict()}"
                )
        else:
            logger.error(f"Column '{value_col}' exists but NO numeric values found!")
            non_numeric_sample = df[value_col].value_counts().head(10).to_dict()
            logger.info(f"Unique non-numeric values: {non_numeric_sample}")
    else:
        logger.warning("Neither 'DataValue' nor 'DataValueAlt' found. Skipping value validation.")

    # ============================================================
    # 4. STRATIFICATION VALIDATION
    # ============================================================
    strat_cols = [col for col in df.columns if col.startswith('Stratification') and not col.endswith('ID')]
    if strat_cols:
        for col in strat_cols[:3]:  # Check first 3 to avoid spam
            non_null = df[col].dropna()
            if len(non_null) > 0:
                unique_strats = non_null.nunique()
                top_strats = non_null.value_counts().head(3).to_dict()
                logger.info(
                    f"{col}: {unique_strats} unique values. "
                    f"Top: {top_strats}"
                )
            else:
                logger.warning(f"Column '{col}' is entirely null/empty.")

    # ============================================================
    # 5. LOCATION VALIDATION
    # ============================================================
    if "LocationAbbr" in df.columns:
        unique_locations = df["LocationAbbr"].dropna().unique()
        logger.info(f"Unique locations: {len(unique_locations)} (e.g., {unique_locations[:5].tolist()})")
        
        # Check for expected US state abbreviations
        us_states = {'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
                     'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
                     'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
                     'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
                     'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'}
        
        non_state_locations = set(unique_locations) - us_states
        if non_state_locations:
            logger.info(
                f"Non-state locations found: {sorted(non_state_locations)} "
                f"(e.g., 'US', 'PR', 'VI', 'DC' are normal)"
            )

    # ============================================================
    # 6. OVERALL DATA COMPLETENESS
    # ============================================================
    total_cells = df.shape[0] * df.shape[1]
    null_cells = df.isnull().sum().sum()
    null_percentage = (null_cells / total_cells) * 100 if total_cells > 0 else 0
    
    logger.info(
        f"Data completeness: {null_cells:,} missing cells / {total_cells:,} total "
        f"({null_percentage:.2f}% null)"
    )
    
    if null_percentage > 30:
        logger.warning(
            f"High null percentage: {null_percentage:.1f}%. "
            f"Consider if stratification creates sparse data."
        )

    # ============================================================
    # 7. DUPLICATE CHECK
    # ============================================================
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        logger.warning(f"Found {duplicates:,} duplicate rows. Check if data loading created duplicates.")


def run_raw_data_validations(df: pd.DataFrame) -> None:
    """
    Run all raw data validations with per-DataValueType bounds.
    """
    logger.info("Starting data validation pipeline...")

    # 1. Schema
    required = [
        "YearStart",
        "LocationAbbr",
        "LocationDesc",
        "Topic",
        "Question",
        "DataValue",
        "LowConfidenceLimit",
        "HighConfidenceLimit",
        "DataValueType",
    ]
    validate_schema(df, required)

    # 2. Not empty
    validate_not_empty(df)

    # 3. Log DataValueType distribution
    if "DataValueType" in df.columns:
        vtype_counts = df["DataValueType"].value_counts()
        logger.info(f"DataValueType distribution:\n{vtype_counts.to_string()}")
    
    # 4. Per-DataValueType bounds for DataValue
    # The validate_value_ranges function will handle per-type validation
    # We just need to set a very broad global bound as fallback
    logger.info("Validating DataValue with per-DataValueType bounds...")
    
    # Broad global bounds - the per-type validation will be stricter
    ranges = {}
    
    # Year columns - strict
    if "YearStart" in df.columns:
        ranges["YearStart"] = (2000, 2030)
    if "YearEnd" in df.columns:
        ranges["YearEnd"] = (2000, 2030)
    
    # DataValue - extremely broad global bound (10 million max)
    # Per-DataValueType validation will catch real issues
    if "DataValue" in df.columns:
        ranges["DataValue"] = (0, 10_000_000)  # 10 million max for counts
    
    # Confidence Limits - broad (rates per 100,000 can be high)
    if "LowConfidenceLimit" in df.columns:
        ranges["LowConfidenceLimit"] = (0, 5000)
    if "HighConfidenceLimit" in df.columns:
        ranges["HighConfidenceLimit"] = (0, 5000)
    
    # Run validations - the function will apply per-type bounds automatically
    for col, bounds in ranges.items():
        if col in df.columns:
            validate_value_ranges(df, {col: bounds})

    # 5. Basic Quality checks (warnings, not fatal)
    validate_basic_quality(df)

    logger.info("All validations passed successfully!")


# ============================================================
# PART 2: POST-CLEANING VALIDATION
# ============================================================

def validate_after_cleaning(df: pd.DataFrame) -> None:
    """
    Validate data after cleaning.
    Checks: no missing values in critical columns, no duplicates, types correct.
    """
    logger.info("Running POST-CLEANING validations...")
    
    # 1. No missing values in critical columns
    critical_cols = ["YearStart", "DataValue", "LocationAbbr"]
    for col in critical_cols:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                logger.warning(f"Column '{col}' still has {null_count:,} nulls after cleaning.")
            else:
                logger.info(f"✅ '{col}' has 0 missing values.")
    
    # 2. No duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        logger.warning(f"Found {duplicates:,} duplicate rows after cleaning.")
    else:
        logger.info("✅ No duplicate rows found.")
    
    # 3. DataValue is numeric
    if "DataValue" in df.columns:
        numeric_values = pd.to_numeric(df["DataValue"], errors="coerce")
        non_numeric_count = numeric_values.isna().sum()
        if non_numeric_count > 0:
            logger.warning(f"DataValue still has {non_numeric_count:,} non-numeric values after cleaning.")
        else:
            logger.info("✅ DataValue is fully numeric.")
    
    # 4. Row count sanity
    if df.shape[0] == 0:
        raise ValueError("DataFrame is empty after cleaning!")
    
    logger.info("✅ Post-cleaning validations passed!\n")


# ============================================================
# PART 3: POST-FEATURE-ENGINEERING VALIDATION
# ============================================================

def validate_ml_ready(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that DataFrame is ready for ML models.
    Call this AFTER feature engineering, BEFORE train/test split.
    
    Returns:
        DataFrame with constant columns removed (if any)
    """
    logger.info("Running ML-READY validations...")
    
    # 1. No non-numeric columns
    non_numeric = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if non_numeric:
        raise ValueError(f"❌ Non-numeric columns found: {non_numeric}")
    logger.info(f"✅ All {df.shape[1]} columns are numeric.")
    
    # 2. No missing values
    null_cols = df.columns[df.isnull().any()].tolist()
    if null_cols:
        null_counts = df[null_cols].isnull().sum().to_dict()
        raise ValueError(f"❌ Missing values found in: {null_counts}")
    logger.info("✅ No missing values.")
    
    # 3. No infinite values
    if np.isinf(df.values).any():
        inf_cols = df.columns[np.isinf(df.values).any(axis=0)].tolist()
        raise ValueError(f"❌ Infinite values found in: {inf_cols}")
    logger.info("✅ No infinite values.")
    
    # 4. Feature count is reasonable
    if df.shape[1] > 1000:
        logger.warning(f"⚠️ High feature count: {df.shape[1]}. Consider feature selection.")
    else:
        logger.info(f"✅ Feature count: {df.shape[1]} (reasonable).")
    
    # 5. Check for constant columns (add no value)
    constant_cols = []
    for col in df.columns:
        if df[col].std() == 0:
            constant_cols.append(col)
    if constant_cols:
        logger.warning(f"Constant columns found: {constant_cols[:5]}... (consider dropping)")
        df.drop(columns=constant_cols, inplace=True)
        logger.info(f"✅ Dropped {len(constant_cols)} constant columns.")
    
    # 6. Memory usage
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    if memory_mb > 500:
        logger.warning(f"⚠️ High memory usage: {memory_mb:.2f} MB. Consider downcasting types.")
    else:
        logger.info(f"✅ Memory usage: {memory_mb:.2f} MB.")
    
    logger.info("✅ ML-ready validations passed!\n")
    return df


# ============================================================
# PART 4: TRAIN/TEST SPLIT VALIDATION
# ============================================================

def validate_train_test_split(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    y_train: pd.Series, 
    y_test: pd.Series
) -> None:
    """
    Validate that train/test split is representative and has no leakage.
    """
    logger.info("Running TRAIN/TEST SPLIT validations...")
    
    # 1. No data leakage (index overlap)
    train_indices = set(X_train.index)
    test_indices = set(X_test.index)
    overlap = train_indices & test_indices
    if overlap:
        raise ValueError(f"❌ Data leakage! {len(overlap)} rows appear in both train and test.")
    logger.info("✅ No data leakage (indices don't overlap).")
    
    # 2. Target distribution similar
    y_train_mean = y_train.mean()
    y_test_mean = y_test.mean()
    diff_pct = abs(y_train_mean - y_test_mean) / max(abs(y_train_mean), 1e-10) * 100
    
    if diff_pct > 10:
        logger.warning(
            f"⚠️ Target distribution differs: train mean={y_train_mean:.4f}, "
            f"test mean={y_test_mean:.4f} ({diff_pct:.1f}% diff)"
        )
    else:
        logger.info(f"✅ Target distribution similar: train={y_train_mean:.4f}, test={y_test_mean:.4f}")
    
    # 3. Split size is reasonable
    train_ratio = len(X_train) / (len(X_train) + len(X_test))
    if not 0.6 <= train_ratio <= 0.9:
        logger.warning(f"⚠️ Unusual train ratio: {train_ratio:.2f} (expected 0.7-0.8)")
    else:
        logger.info(f"✅ Train ratio: {train_ratio:.2f}")
    
    logger.info("✅ Train/test split validations passed!\n")


# ============================================================
# PART 5: SCALING VALIDATION
# ============================================================

def validate_scaling(
    X_train_scaled: pd.DataFrame, 
    X_test_scaled: pd.DataFrame, 
    scaler_type: str = "StandardScaler"
) -> None:
    """
    Validate that scaling worked correctly.
    """
    logger.info("Running SCALING validations...")
    
    if scaler_type == "StandardScaler":
        # Mean should be ~0, Std should be ~1
        means = X_train_scaled.mean()
        stds = X_train_scaled.std()
        
        mean_off = means[abs(means) > 0.01].tolist()
        std_off = stds[(stds < 0.9) | (stds > 1.1)].tolist()
        
        if mean_off:
            logger.warning(f"⚠️ Some features have mean != 0: {len(mean_off)} features")
        else:
            logger.info("✅ All features have mean ~0.")
        
        if std_off:
            logger.warning(f"⚠️ Some features have std != 1: {len(std_off)} features")
        else:
            logger.info("✅ All features have std ~1.")
    
    elif scaler_type == "MinMaxScaler":
        # Min should be ~0, Max should be ~1
        mins = X_train_scaled.min()
        maxs = X_train_scaled.max()
        
        min_off = mins[(mins < -0.01) | (mins > 0.01)].tolist()
        max_off = maxs[(maxs < 0.99) | (maxs > 1.01)].tolist()
        
        if min_off:
            logger.warning(f"⚠️ Some features have min != 0: {len(min_off)} features")
        else:
            logger.info("✅ All features have min ~0.")
        
        if max_off:
            logger.warning(f"⚠️ Some features have max != 1: {len(max_off)} features")
        else:
            logger.info("✅ All features have max ~1.")
    
    # Check that test data is scaled consistently
    test_min = X_test_scaled.min().min()
    test_max = X_test_scaled.max().max()
    
    if test_min < -5 or test_max > 5:
        logger.warning(f"⚠️ Test data has extreme values: min={test_min:.2f}, max={test_max:.2f}")
    else:
        logger.info(f"✅ Test data within expected range: min={test_min:.2f}, max={test_max:.2f}")
    
    logger.info("✅ Scaling validations passed!\n")


# ============================================================
# PART 6: PREDICTION VALIDATION
# ============================================================

def validate_predictions(
    predictions: np.ndarray, 
    y_test: np.ndarray, 
    task_type: str = "regression"
) -> None:
    """
    Validate that predictions make sense.
    """
    logger.info("Running PREDICTION validations...")
    
    # 1. No NaN or Inf
    if np.isnan(predictions).any():
        raise ValueError("❌ Predictions contain NaN values!")
    if np.isinf(predictions).any():
        raise ValueError("❌ Predictions contain Inf values!")
    logger.info("✅ No NaN or Inf in predictions.")
    
    # 2. Predictions in reasonable range
    if task_type == "regression":
        pred_min, pred_max = predictions.min(), predictions.max()
        target_min, target_max = y_test.min(), y_test.max()
        target_range = target_max - target_min
        
        # Check if predictions are wildly outside target range
        if pred_min < target_min - 2 * target_range:
            logger.warning(f"⚠️ Predictions lower than target min: {pred_min:.4f} vs {target_min:.4f}")
        if pred_max > target_max + 2 * target_range:
            logger.warning(f"⚠️ Predictions higher than target max: {pred_max:.4f} vs {target_max:.4f}")
        logger.info(f"✅ Prediction range: {pred_min:.4f} - {pred_max:.4f}")
    
    elif task_type == "classification":
        # For binary classification, probabilities should be between 0 and 1
        if (predictions < 0).any() or (predictions > 1).any():
            logger.warning("⚠️ Probabilities outside [0, 1] range!")
        logger.info(f"✅ Probability range: {predictions.min():.4f} - {predictions.max():.4f}")
    
    logger.info("✅ Prediction validations passed!\n")


# ============================================================
# PART 7: FULL PIPELINE VALIDATION (Legacy Compatibility)
# ============================================================

def run_all_validations(df: pd.DataFrame) -> None:
    """
    Legacy compatibility wrapper for run_raw_data_validations.
    Deprecated: Use run_raw_data_validations() instead.
    """
    logger.warning("run_all_validations() is deprecated. Use run_raw_data_validations() instead.")
    run_raw_data_validations(df)