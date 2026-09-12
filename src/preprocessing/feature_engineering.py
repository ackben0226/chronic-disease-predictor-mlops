"""
Feature engineering
Creates the derived features for the Chronic Disease Indicator dataset.
Handles demographics, geography, uncertainty, and missingness.
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# --- Helper: US Regions (Geography) ---
def _get_us_region(state_abbr: str) -> str:
    """Map US state abbreviations to Census Regions."""
    if pd.isna(state_abbr):
        return 'Unknown'
    regions = {
        'Northeast': ['CT', 'ME', 'MA', 'NH', 'RI', 'VT', 'NJ', 'NY', 'PA'],
        'Midwest': ['IL', 'IN', 'MI', 'OH', 'WI', 'IA', 'KS', 'MN', 'MO', 'NE', 'ND', 'SD'],
        'South': ['DE', 'FL', 'GA', 'MD', 'NC', 'SC', 'VA', 'DC', 'WV', 'AL', 'KY', 'MS', 'TN', 'AR', 'LA', 'OK', 'TX'],
        'West': ['AZ', 'CO', 'ID', 'MT', 'NV', 'NM', 'UT', 'WY', 'AK', 'CA', 'HI', 'OR', 'WA']
    }
    for region, states in regions.items():
        if state_abbr in states:
            return region
    return 'Unknown'


# --- Helper: State Centroids (for missing coordinates) ---
def _get_state_centroid(state_abbr: str) -> tuple:
    """Get approximate centroid coordinates for a US state/territory."""
    centroids = {
        # US States
        'AL': (32.8067, -86.7911), 'AK': (61.3707, -152.4044), 'AZ': (33.7298, -111.4312),
        'AR': (34.9697, -92.3731), 'CA': (36.1162, -119.6816), 'CO': (39.0598, -105.3111),
        'CT': (41.5978, -72.7554), 'DE': (38.9108, -75.5277), 'FL': (27.7663, -81.6868),
        'GA': (32.3294, -83.1137), 'HI': (21.0943, -157.4983), 'ID': (44.2405, -114.4788),
        'IL': (40.3495, -88.9861), 'IN': (39.8934, -86.1346), 'IA': (42.0115, -93.2105),
        'KS': (38.5266, -96.7265), 'KY': (37.6681, -84.6701), 'LA': (31.1695, -91.8678),
        'ME': (44.6939, -69.3819), 'MD': (39.0639, -76.8021), 'MA': (42.2302, -71.5301),
        'MI': (43.3266, -84.5361), 'MN': (45.6945, -93.9002), 'MS': (32.7416, -89.6787),
        'MO': (38.4561, -92.2884), 'MT': (46.9219, -110.4544), 'NE': (41.1254, -98.2681),
        'NV': (38.3135, -117.0558), 'NH': (43.4525, -71.5639), 'NJ': (40.2989, -74.5210),
        'NM': (34.8405, -106.2485), 'NY': (42.1657, -74.9481), 'NC': (35.6301, -79.8064),
        'ND': (47.5289, -99.7840), 'OH': (40.3888, -82.7649), 'OK': (35.5653, -96.9289),
        'OR': (44.5720, -122.0709), 'PA': (40.5908, -77.2098), 'RI': (41.6809, -71.5118),
        'SC': (33.8569, -80.9450), 'SD': (44.2998, -99.4388), 'TN': (35.7478, -86.6923),
        'TX': (31.0545, -97.5635), 'UT': (40.1500, -111.8624), 'VT': (44.0459, -72.7107),
        'VA': (37.7693, -78.1693), 'WA': (47.4009, -121.4905), 'WV': (38.4912, -80.9545),
        'WI': (44.2685, -89.6165), 'WY': (42.7560, -107.3025),
        # Territories and national
        'DC': (38.9072, -77.0369), 'PR': (18.2208, -66.5901), 'VI': (18.3358, -64.8963),
        'GU': (13.4443, 144.7937), 'AS': (-14.2710, -170.1322), 'MP': (15.0979, 145.6739),
        'US': (39.8283, -98.5795), '': (39.8283, -98.5795)
    }
    return centroids.get(state_abbr, (39.8283, -98.5795))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline for Chronic Disease Indicators.

    Steps:
    1. Convert DataValue to numeric FIRST (critical!)
    2. Drop raw ID columns (they add noise and cause data leakage).
    3. Extract Latitude/Longitude from GeoLocation with fallback.
    4. Map state abbreviations to US_Region.
    5. Create Confidence Interval Width (measure of data reliability).
    6. Create Missing Data Flag (so the model knows which values are imputed).
    7. One-hot encode Demographics (Stratification1) e.g., Overall, Male, Female.
    8. One-hot encode Topic and Question (for ML compatibility).

    Args:
        df: Cleaned DataFrame from cleaner.py.

    Returns:
        DataFrame with engineered features, ready for ML.
    """
    logger.info("Starting feature engineering...")
    original_cols = df.shape[1]

    # ============================================================
    # 0. CONVERT DATAVALUE TO NUMERIC FIRST (CRITICAL!)
    # ============================================================
    if 'DataValue' in df.columns:
        # Convert to numeric, coerce non-numeric to NaN
        df['DataValue'] = pd.to_numeric(df['DataValue'], errors='coerce')
        
        # Log conversion results
        valid_count = df['DataValue'].notna().sum()
        total_count = len(df)
        logger.info(
            f"DataValue converted to numeric: {valid_count:,}/{total_count:,} valid "
            f"({valid_count/total_count*100:.1f}%)"
        )
        
        # Create missing flag BEFORE imputation
        df['DataValue_Missing_Flag'] = df['DataValue'].isna().astype(int)
        missing_count = df['DataValue_Missing_Flag'].sum()
        logger.info(f"Created 'DataValue_Missing_Flag'. Missing count: {missing_count:,}")
        
        # Impute missing values - use group medians by DataValueType if available
        if 'DataValueType' in df.columns:
            def safe_median_imputation(group):
                """Impute with median, return original if all NaN."""
                if group.isna().all():
                    return group
                return group.fillna(group.median())
            
            # Fill with group median
            df['DataValue'] = df.groupby('DataValueType')['DataValue'].transform(safe_median_imputation)
            
            # Check if any are still NaN (groups with all NaN)
            remaining_nulls = df['DataValue'].isna().sum()
            if remaining_nulls > 0:
                global_median = df['DataValue'].median()
                df['DataValue'] = df['DataValue'].fillna(global_median)
                logger.warning(
                    f"Some DataValue groups all NaN. Filled {remaining_nulls} remaining with global median: {global_median:.2f}"
                )
            else:
                logger.info("Filled DataValue missing with group medians by DataValueType")
        else:
            # Simple global median imputation
            global_median = df['DataValue'].median()
            df['DataValue'] = df['DataValue'].fillna(global_median)
            logger.info(f"Filled DataValue missing with global median: {global_median:.2f}")

    # ============================================================
    # 1. DROP USELESS ID COLUMNS (Prevents overfitting)
    # ============================================================
    cols_to_drop = [
        'ResponseID', 'LocationID', 'TopicID', 'QuestionID',
        'DataValueTypeID', 'StratificationCategoryID1', 'StratificationID1',
        'StratificationCategoryID2', 'StratificationID2',
        'StratificationCategoryID3', 'StratificationID3',
        'DataValueFootnoteSymbol', 'DatavalueFootnote', 'DataSource'
    ]
    existing_drops = [col for col in cols_to_drop if col in df.columns]
    if existing_drops:
        df.drop(columns=existing_drops, inplace=True)
        logger.debug(f"Dropped ID/Footnote columns: {existing_drops}")

    # ============================================================
    # 2. STATISTICAL UNCERTAINTY (Confidence Interval Width)
    # ============================================================
    if 'HighConfidenceLimit' in df.columns and 'LowConfidenceLimit' in df.columns:
        # Convert to numeric (in case they're still strings)
        df['HighConfidenceLimit'] = pd.to_numeric(df['HighConfidenceLimit'], errors='coerce')
        df['LowConfidenceLimit'] = pd.to_numeric(df['LowConfidenceLimit'], errors='coerce')
        
        df['ConfidenceIntervalWidth'] = df['HighConfidenceLimit'] - df['LowConfidenceLimit']
        logger.info(
            f"Created 'ConfidenceIntervalWidth'. "
            f"Avg: {df['ConfidenceIntervalWidth'].mean():.2f}, "
            f"Min: {df['ConfidenceIntervalWidth'].min():.2f}, "
            f"Max: {df['ConfidenceIntervalWidth'].max():.2f}"
        )

    # ============================================================
    # 3. GEOGRAPHY: Extract Lat/Long & Region (WITH FIX)
    # ============================================================
    if 'GeoLocation' in df.columns:
        # Extract Latitude and Longitude
        df['Latitude'] = df['GeoLocation'].str.extract(r'\(([^,]+),').astype(float)
        df['Longitude'] = df['GeoLocation'].str.extract(r',\s*([^)]+)\)').astype(float)
        logger.debug("Extracted Latitude and Longitude.")
        
        # --- FIX: Handle missing coordinates ---
        lat_nulls = df['Latitude'].isna().sum()
        lon_nulls = df['Longitude'].isna().sum()
        
        if lat_nulls > 0 or lon_nulls > 0:
            logger.warning(f"Found {lat_nulls:,} missing Latitude and {lon_nulls:,} missing Longitude")
            
            # Fill with state centroids using LocationAbbr
            if 'LocationAbbr' in df.columns:
                filled_count = 0
                for idx, row in df[df['Latitude'].isna() | df['Longitude'].isna()].iterrows():
                    state = row.get('LocationAbbr', '')
                    lat, lon = _get_state_centroid(state)
                    if pd.isna(df.loc[idx, 'Latitude']):
                        df.loc[idx, 'Latitude'] = lat
                    if pd.isna(df.loc[idx, 'Longitude']):
                        df.loc[idx, 'Longitude'] = lon
                    filled_count += 1
                logger.info(f"Filled coordinates for {filled_count:,} rows using state centroids")
            
            # Final fallback - fill any remaining NaN with US centroid
            df['Latitude'] = df['Latitude'].fillna(39.8283)
            df['Longitude'] = df['Longitude'].fillna(-98.5795)
            
            # Verify no remaining NaN
            remaining_lat = df['Latitude'].isna().sum()
            remaining_lon = df['Longitude'].isna().sum()
            if remaining_lat > 0 or remaining_lon > 0:
                logger.warning(f"⚠️ Still have {remaining_lat:,} missing Latitude and {remaining_lon:,} missing Longitude")
        
        df.drop(columns=['GeoLocation'], inplace=True)

        # Map state abbreviations to US Region
        if 'LocationAbbr' in df.columns:
            df['US_Region'] = df['LocationAbbr'].apply(_get_us_region)
            logger.info("Created 'US_Region' feature.")
            
            # One-hot encode US_Region
            region_dummies = pd.get_dummies(df['US_Region'], prefix='Region')
            df = pd.concat([df, region_dummies], axis=1)
            logger.info(f"One-hot encoded 'US_Region'. Added {region_dummies.shape[1]} region features.")
            df.drop(columns=['US_Region'], inplace=True)

    # ============================================================
    # 4. DEMOGRAPHICS: Encode Stratification1
    # ============================================================
    if 'Stratification1' in df.columns:
        df['Stratification1'] = df['Stratification1'].fillna('Unknown').astype(str)
        strat_dummies = pd.get_dummies(df['Stratification1'], prefix='Demo')
        df = pd.concat([df, strat_dummies], axis=1)
        logger.info(f"One-hot encoded 'Stratification1'. Added {strat_dummies.shape[1]} demographic features.")
        df.drop(columns=['Stratification1'], inplace=True)

    # ============================================================
    # 5. CATEGORICAL ENCODING: Topic & Question
    # ============================================================
    if 'Topic' in df.columns:
        topic_dummies = pd.get_dummies(df['Topic'], prefix='Topic')
        df = pd.concat([df, topic_dummies], axis=1)
        logger.debug(f"One-hot encoded 'Topic'. Added {topic_dummies.shape[1]} columns.")
        df.drop(columns=['Topic'], inplace=True)

    if 'Question' in df.columns:
        # If too many unique questions, keep only top 20
        if df['Question'].nunique() > 20:
            top_questions = df['Question'].value_counts().head(20).index
            df['Question_Top'] = df['Question'].apply(
                lambda x: x if x in top_questions else 'Other'
            )
            q_dummies = pd.get_dummies(df['Question_Top'], prefix='Question')
            df = pd.concat([df, q_dummies], axis=1)
            logger.debug(
                f"One-hot encoded top 20 'Question' values. "
                f"Added {q_dummies.shape[1]} columns. "
                f"(Original had {df['Question'].nunique()} unique values)"
            )
            df.drop(columns=['Question', 'Question_Top'], inplace=True)
        else:
            q_dummies = pd.get_dummies(df['Question'], prefix='Question')
            df = pd.concat([df, q_dummies], axis=1)
            logger.debug(f"One-hot encoded 'Question'. Added {q_dummies.shape[1]} columns.")
            df.drop(columns=['Question'], inplace=True)

    # ============================================================
    # 6. TEMPORAL FEATURE (Time trends)
    # ============================================================
    if 'YearStart' in df.columns:
        df['YearStart'] = pd.to_numeric(df['YearStart'], errors='coerce')
        df['YearsSince2020'] = df['YearStart'] - 2020
        logger.debug(f"Created 'YearsSince2020' feature. Range: {df['YearsSince2020'].min():.0f} - {df['YearsSince2020'].max():.0f}")
        
        # Drop YearEnd if exists
        if 'YearEnd' in df.columns:
            df.drop(columns=['YearEnd'], inplace=True)

    # ============================================================
    # 7. CLEAN UP STRING COLUMNS THAT WEREN'T ENCODED
    # ============================================================
    # Drop location columns (already encoded via region)
    if 'LocationAbbr' in df.columns:
        df.drop(columns=['LocationAbbr'], inplace=True)
    if 'LocationDesc' in df.columns:
        df.drop(columns=['LocationDesc'], inplace=True)

    # Drop other categorical columns that are now encoded
    drop_meta = [
        'StratificationCategory1', 'StratificationCategory2', 'Stratification2',
        'StratificationCategory3', 'Stratification3', 'DataValueUnit', 'DataValueType'
    ]
    for col in drop_meta:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # ============================================================
    # 8. ENSURE ALL COLUMNS ARE NUMERIC (for ML models)
    # ============================================================
    # Check for any remaining non-numeric columns
    non_numeric = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if non_numeric:
        logger.warning(f"Remaining non-numeric columns: {non_numeric}")
        # Drop any remaining object columns (should be none)
        df.drop(columns=non_numeric, inplace=True)

    # ============================================================
    # 9. FINAL LOGGING
    # ============================================================
    logger.info(
        f"Feature engineering complete. "
        f"Total columns: {df.shape[1]} "
        f"(added {df.shape[1] - original_cols} new features). "
        f"Rows: {df.shape[0]:,}"
    )
    
    # Log memory usage
    memory_usage = df.memory_usage(deep=True).sum() / 1024**2
    logger.info(f"DataFrame memory usage: {memory_usage:.2f} MB")

    # --- Final validation: check for NaN values ---
    null_cols = df.columns[df.isnull().any()].tolist()
    if null_cols:
        logger.warning(f"⚠️ Columns with NaN after feature engineering: {null_cols}")
        for col in null_cols:
            null_count = df[col].isna().sum()
            logger.warning(f"  {col}: {null_count:,} NaN values")

    return df