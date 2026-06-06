"""Data loading utilities for train/validation split with stratification."""

from typing import Dict, List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from ..utils.logging import get_logger

logger = get_logger(__name__)


def load_data(
    train_path: str,
    test_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load training and test data from CSV files.
    
    Args:
        train_path: Path to training CSV
        test_path: Path to test CSV
        
    Returns:
        Tuple of (train_df, test_df)
    """
    logger.info(f"Loading training data from {train_path}")
    train_df = pd.read_csv(train_path)
    
    logger.info(f"Loading test data from {test_path}")
    test_df = pd.read_csv(test_path)
    
    logger.info(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    
    return train_df, test_df


def create_label_column(df: pd.DataFrame) -> pd.DataFrame:
    """Create integer label column from one-hot encoded labels.
    
    Args:
        df: DataFrame with winner_model_a, winner_model_b, winner_tie columns
        
    Returns:
        DataFrame with added 'label' column (0=A, 1=B, 2=Tie)
    """
    df = df.copy()
    
    def get_label(row):
        if row.get("winner_model_a", 0) == 1:
            return 0
        elif row.get("winner_model_b", 0) == 1:
            return 1
        else:
            return 2
    
    df["label"] = df.apply(get_label, axis=1)
    
    return df


def df_to_list(df: pd.DataFrame) -> List[Dict]:
    """Convert DataFrame to list of dictionaries.
    
    Args:
        df: Input DataFrame
        
    Returns:
        List of row dictionaries
    """
    return df.to_dict("records")


def stratified_split(
    train_df: pd.DataFrame,
    validation_ratio: float = 0.1,
    random_state: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """Perform stratified train/validation split.
    
    IMPORTANT: Split must be done BEFORE applying data augmentation
    to prevent data leakage.
    
    Args:
        train_df: Full training DataFrame
        validation_ratio: Fraction for validation
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_list, val_list)
    """
    # Create label column for stratification
    df_with_label = create_label_column(train_df)
    
    # Log original distribution
    label_counts = df_with_label["label"].value_counts().sort_index()
    logger.info(f"Original label distribution: {label_counts.to_dict()}")
    
    # Perform stratified split
    train_split, val_split = train_test_split(
        df_with_label,
        test_size=validation_ratio,
        stratify=df_with_label["label"],
        random_state=random_state,
    )
    
    # Log split distribution
    train_counts = train_split["label"].value_counts().sort_index()
    val_counts = val_split["label"].value_counts().sort_index()
    logger.info(f"Train split: {len(train_split)} samples, labels: {train_counts.to_dict()}")
    logger.info(f"Val split: {len(val_split)} samples, labels: {val_counts.to_dict()}")
    
    # Convert to list of dicts (drop the temporary label column)
    train_list = df_to_list(train_split.drop(columns=["label"], errors="ignore"))
    val_list = df_to_list(val_split.drop(columns=["label"], errors="ignore"))
    
    return train_list, val_list


def load_and_split(
    train_path: str,
    test_path: str,
    validation_ratio: float = 0.1,
    random_state: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Load data and perform stratified split.
    
    Args:
        train_path: Path to training CSV
        test_path: Path to test CSV
        validation_ratio: Fraction for validation
        random_state: Random seed
        
    Returns:
        Tuple of (train_list, val_list, test_list)
    """
    # Load data
    train_df, test_df = load_data(train_path, test_path)
    
    # Split training data
    train_list, val_list = stratified_split(
        train_df,
        validation_ratio=validation_ratio,
        random_state=random_state,
    )
    
    # Convert test data to list
    test_list = df_to_list(test_df)
    
    logger.info(f"Final splits: Train={len(train_list)}, Val={len(val_list)}, Test={len(test_list)}")
    
    return train_list, val_list, test_list
