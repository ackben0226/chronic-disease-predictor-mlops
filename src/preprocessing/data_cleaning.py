"""
Data cleaning module.
Preprocesses raw CDC data with proper outlier handling and imputation.
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess data to remove data inconsistencies,
    including missing data, white space, duplicates, outliers, etc.

    Args:
        df: Raw pandas DataFrame to be cleaned.

    Returns:
        Cleaned pandas DataFrame.
    """
    logger.info("=" * 60)
    logger.info("STARTING DATA CLEANING PIPELINE")
    logger.info("=" * 60)
    
    # Make a copy to avoid modifying original
    df = df.copy()
    initial_rows = df.shape[0]
    logger.info(f"Initial shape: {initial_rows:,} rows, {df.shape[1]} columns")

    # --- 1. Clean Column Names ---
    df.columns = df.columns.str.strip()
    logger.debug("Stripped whitespace from column names.")

    # --- 2. Strip Whitespace from String VALUES ---
    string_cols = df.select_dtypes(include=["object"]).columns
    for col in string_cols:
        # First, fill NaN with 'unknown' BEFORE converting to string
        df[col] = df[col].fillna('unknown')
        # Then convert to string and strip
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace(r"^\s*$", "Unknown", regex=True)
    if len(string_cols) > 0:
        logger.debug(f"Stripped whitespace from {len(string_cols)} string columns.")

    # --- 3. CRITICAL: CONVERT DATAVALUE TO NUMERIC FIRST ---
    # This MUST happen before any median calculations
    if "DataValue" in df.columns:
        logger.info("Converting DataValue to numeric...")
        df['DataValue'] = pd.to_numeric(df['DataValue'], errors='coerce')
        
        valid_count = df['DataValue'].notna().sum()
        total_count = len(df)
        valid_pct = valid_count / total_count * 100 if total_count > 0 else 0
        logger.info(
            f"DataValue converted: {valid_count:,}/{total_count:,} valid "
            f"({valid_pct:.1f}%)"
        )
        
        # Log non-numeric values for debugging (if any)
        if valid_count < total_count:
            non_numeric = df[df['DataValue'].isna()]
            if 'DataValueType' in non_numeric.columns:
                non_numeric_types = non_numeric['DataValueType'].value_counts().head(5)
                logger.info(f"Non-numeric DataValue types: {non_numeric_types.to_dict()}")

    # --- 4. Convert Other Numeric Columns ---
    for col in ['YearStart', 'YearEnd', 'LowConfidenceLimit', 'HighConfidenceLimit']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            null_count = df[col].isna().sum()
            if null_count > 0:
                logger.warning(f"Column '{col}' has {null_count:,} non-numeric values converted to NaN.")

    # --- 5. Handle Missing Values (Smart Imputation) ---
    total_nulls = df.isnull().sum().sum()
    if total_nulls > 0:
        logger.warning(f"Found {total_nulls:,} missing values. Applying smart imputation...")
        
        # 5a. DataValue: impute by DataValueType (now that it's numeric!)
        if "DataValueType" in df.columns and "DataValue" in df.columns:
            logger.info("Imputing DataValue by DataValueType...")
            
            # Compute group medians (handle groups with all NaN)
            def safe_median(group):
                """Compute median, return NaN if group is all NaN."""
                if group.isna().all():
                    return np.nan
                return group.median()
            
            try:
                # Fill with group median
                df['DataValue'] = df.groupby('DataValueType')['DataValue'].transform(
                    lambda x: x.fillna(safe_median(x))
                )
                
                # Fill any remaining NaN with global median
                remaining_nulls = df['DataValue'].isna().sum()
                if remaining_nulls > 0:
                    global_median = df['DataValue'].median()
                    df['DataValue'] = df['DataValue'].fillna(global_median)
                    logger.warning(
                        f"Filled {remaining_nulls:,} remaining DataValue NaN with "
                        f"global median: {global_median:.2f}"
                    )
                else:
                    logger.info("✅ DataValue imputation by DataValueType complete.")
                    
            except Exception as e:
                logger.error(f"Error during DataValue imputation: {e}")
                # Fallback: use global median
                global_median = df['DataValue'].median()
                df['DataValue'] = df['DataValue'].fillna(global_median)
                logger.warning(f"Fallback: Filled DataValue with global median: {global_median:.2f}")
        
        # 5b. Other numeric columns: fill with median
        numeric_cols = df.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            if col != "DataValue" and df[col].isnull().any():
                col_median = df[col].median()
                df[col].fillna(col_median, inplace=True)
                logger.debug(f"Filled nulls in numeric column '{col}' with median: {col_median:.2f}")
        
        # 5c. String columns: fill with "unknown"
        for col in string_cols:
            if col in df.columns and df[col].isnull().any():
                df[col].fillna("unknown", inplace=True)
                logger.debug(f"Filled nulls in string column '{col}' with 'unknown'.")
    else:
        logger.info("No missing values found.")

    # --- 6. Handle Outliers ---
    logger.info("Detecting and handling outliers...")
    
    numeric_cols = df.select_dtypes(include=["number"]).columns
    skip_cols = ['LocationID', 'YearStart', 'YearEnd', 'DataValue_Missing_Flag']
    numeric_cols = [col for col in numeric_cols if col not in skip_cols]
    
    outlier_count = 0
    for col in numeric_cols:
        # Skip if column has all NaN (shouldn't happen after imputation)
        if df[col].isna().all():
            continue
            
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Skip if IQR is 0 (constant column)
        if IQR == 0:
            continue
        
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_count_col = outliers.sum()
        
        if outlier_count_col > 0:
            outlier_pct = outlier_count_col / len(df) * 100
            logger.warning(
                f"Column '{col}': {outlier_count_col:,} outliers ({outlier_pct:.2f}%) "
                f"outside [{lower_bound:.2f}, {upper_bound:.2f}]"
            )
            
            # For DataValue, cap extreme outliers (but don't remove)
            if col == "DataValue":
                upper_cap = df[col].quantile(0.995)
                lower_cap = df[col].quantile(0.005)
                df[col] = df[col].clip(lower=lower_cap, upper=upper_cap)
                logger.info(
                    f"Capped DataValue at [0.5th: {lower_cap:.2f}, "
                    f"99.5th: {upper_cap:.2f}]"
                )
            elif col in ["LowConfidenceLimit", "HighConfidenceLimit"]:
                upper_cap = df[col].quantile(0.99)
                df[col] = df[col].clip(upper=upper_cap)
                logger.info(f"Capped {col} at 99th percentile: {upper_cap:.2f}")
            else:
                lower_cap = df[col].quantile(0.01)
                upper_cap = df[col].quantile(0.99)
                df[col] = df[col].clip(lower=lower_cap, upper=upper_cap)
                logger.debug(f"Capped '{col}' at 1st and 99th percentiles.")
            
            outlier_count += outlier_count_col
    
    if outlier_count > 0:
        logger.info(f"Handled {outlier_count:,} outliers total.")
    else:
        logger.info("No outliers detected.")

    # --- 7. Remove Duplicate Rows ---
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        logger.warning(f"Removed {duplicates:,} duplicate rows.")
        df.drop_duplicates(inplace=True)
    else:
        logger.info("No duplicate rows found.")

    # --- 8. Date/Time Validation ---
    if 'YearStart' in df.columns:
        invalid_years = df[(df['YearStart'] < 1900) | (df['YearStart'] > 2030)]
        if len(invalid_years) > 0:
            logger.warning(f"Found {len(invalid_years):,} rows with invalid YearStart values.")
            df['YearStart'] = df['YearStart'].clip(1900, 2030)
    
    if 'YearEnd' in df.columns:
        invalid_years = df[(df['YearEnd'] < 1900) | (df['YearEnd'] > 2030)]
        if len(invalid_years) > 0:
            logger.warning(f"Found {len(invalid_years):,} rows with invalid YearEnd values.")
            df['YearEnd'] = df['YearEnd'].clip(1900, 2030)

    # --- 9. Flag Suspicious Rows ---
    logger.info("Creating quality flags...")
    
    if 'DataValue' in df.columns:
        data_value_99 = df['DataValue'].quantile(0.99)
        df['DataValue_Extreme_Flag'] = (df['DataValue'] > data_value_99).astype(int)
        extreme_count = df['DataValue_Extreme_Flag'].sum()
        logger.info(f"Created 'DataValue_Extreme_Flag'. {extreme_count:,} rows flagged as extreme.")
    
    if 'LowConfidenceLimit' in df.columns and 'HighConfidenceLimit' in df.columns:
        df['CI_Missing_Flag'] = (
            df['LowConfidenceLimit'].isna() | df['HighConfidenceLimit'].isna()
        ).astype(int)
        ci_missing = df['CI_Missing_Flag'].sum()
        logger.info(f"Created 'CI_Missing_Flag'. {ci_missing:,} rows have missing confidence intervals.")

    # --- 10. Final Data Type Validation ---
    # Ensure all numeric columns are actually float/int
    for col in ['DataValue', 'YearStart', 'YearEnd', 'LowConfidenceLimit', 'HighConfidenceLimit']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isna().any():
                df[col].fillna(df[col].median(), inplace=True)

    # --- 11. Final Report ---
    final_rows = df.shape[0]
    rows_removed = initial_rows - final_rows
    final_nulls = df.isnull().sum().sum()
    
    logger.info("=" * 60)
    logger.info("CLEANING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Rows: {initial_rows:,} → {final_rows:,} (removed {rows_removed:,})")
    logger.info(f"  Columns: {df.shape[1]}")
    logger.info(f"  Missing values remaining: {final_nulls:,}")
    logger.info(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    logger.info("=" * 60)
    
    return df


# ============================================================
# HELPER: Targeted Outlier Removal
# ============================================================

def remove_outliers_by_group(
    df: pd.DataFrame, 
    col: str, 
    group_col: str = "DataValueType",
    method: str = "iqr",
    threshold: float = 3.0
) -> pd.DataFrame:
    """
    Remove outliers within specific groups (e.g., by DataValueType).
    
    Args:
        df: Input DataFrame
        col: Column to check for outliers
        group_col: Column to group by
        method: 'iqr' or 'zscore'
        threshold: Threshold for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    logger.info(f"Removing outliers from '{col}' by '{group_col}'...")
    
    original_len = len(df)
    mask = pd.Series(True, index=df.index)
    
    for group, group_df in df.groupby(group_col):
        if len(group_df) < 10:  # Skip small groups
            continue
            
        values = group_df[col].dropna()
        if len(values) == 0:
            continue
            
        if method == "iqr":
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            group_mask = (group_df[col] >= lower) & (group_df[col] <= upper)
            
        elif method == "zscore":
            try:
                from scipy import stats
                z_scores = np.abs(stats.zscore(values))
                group_mask = z_scores <= threshold
                # Align mask with original data
                group_mask = pd.Series(group_mask.values, index=values.index)
            except ImportError:
                logger.warning("scipy not available, skipping zscore method")
                continue
        
        mask.loc[group_df.index] &= group_mask
    
    df_cleaned = df[mask].copy()
    removed = original_len - len(df_cleaned)
    
    if removed > 0:
        logger.info(f"Removed {removed:,} outlier rows ({removed/original_len*100:.2f}%)")
    else:
        logger.info("No outliers removed.")
    
    return df_cleaned


# ============================================================
# HELPER: Quality Report
# ============================================================

def generate_cleaning_report(df_before: pd.DataFrame, df_after: pd.DataFrame) -> Dict:
    """
    Generate a detailed report of what changed during cleaning.
    
    Returns:
        Dictionary with cleaning statistics
    """
    report = {
        "rows_before": len(df_before),
        "rows_after": len(df_after),
        "rows_removed": len(df_before) - len(df_after),
        "columns_before": len(df_before.columns),
        "columns_after": len(df_after.columns),
        "nulls_before": df_before.isnull().sum().sum(),
        "nulls_after": df_after.isnull().sum().sum(),
        "duplicates_removed": df_before.duplicated().sum() - df_after.duplicated().sum(),
        "outliers_handled": 0
    }
    
    # Count outlier handling (if flags exist)
    if 'DataValue_Extreme_Flag' in df_after.columns:
        report['outliers_handled'] = df_after['DataValue_Extreme_Flag'].sum()
    
    return report