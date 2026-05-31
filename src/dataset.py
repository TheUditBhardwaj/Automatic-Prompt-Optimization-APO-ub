import os
import random
import json
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

class DatasetItem:
    """Represents a single PDF and its corresponding gold annotation."""
    def __init__(self, name: str, pdf_path: str, gold_path: str):
        self.name = name
        self.pdf_path = pdf_path
        self.gold_path = gold_path

    def load_gold_json(self) -> Dict[str, Any]:
        """Loads and returns the gold JSON annotation."""
        with open(self.gold_path, 'r') as f:
            return json.load(f)

    def __repr__(self):
        return f"DatasetItem(name={self.name}, pdf={self.pdf_path}, gold={self.gold_path})"


def load_dataset(raw_dir: str, gold_dir: str) -> List[DatasetItem]:
    """Finds all PDF files in raw_dir and matches them with annotations in gold_dir.
    
    Ensures file matching is exact by filename base.
    """
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")
    if not os.path.isdir(gold_dir):
        raise FileNotFoundError(f"Gold directory does not exist: {gold_dir}")

    items = []
    # Sort files to guarantee deterministic starting order before shuffle
    all_files = sorted(os.listdir(raw_dir))
    pdf_files = [f for f in all_files if f.lower().endswith('.pdf')]

    for pdf_file in pdf_files:
        base_name = os.path.splitext(pdf_file)[0]
        
        pdf_path = os.path.join(raw_dir, pdf_file)
        
        # Check for either .gold.json or .json extensions
        gold_path_options = [
            os.path.join(gold_dir, f"{base_name}.gold.json"),
            os.path.join(gold_dir, f"{base_name}.json")
        ]
        
        gold_path = None
        for option in gold_path_options:
            if os.path.exists(option):
                gold_path = option
                break
                
        if not gold_path:
            logger.warning(f"Skipping {pdf_file}: Gold annotation not found in {gold_dir} (tried .gold.json and .json)")
            continue

        items.append(DatasetItem(
            name=base_name,
            pdf_path=pdf_path,
            gold_path=gold_path
        ))
        
    logger.info(f"Loaded {len(items)} matched dataset items.")
    return items


def split_dataset(
    items: List[DatasetItem], 
    train_ratio: float, 
    val_ratio: float, 
    test_ratio: float, 
    seed: int
) -> Tuple[List[DatasetItem], List[DatasetItem], List[DatasetItem]]:
    """Splits a dataset deterministically into train, validation, and test sets.
    
    Args:
        items: List of DatasetItem objects.
        train_ratio: Ratio of training items (e.g. 0.6)
        val_ratio: Ratio of validation items (e.g. 0.2)
        test_ratio: Ratio of test items (e.g. 0.2)
        seed: Random seed for shuffling.
        
    Returns:
        A tuple of (train_items, val_items, test_items)
    """
    # Create a local Random instance with the specified seed to avoid global state mutation
    rng = random.Random(seed)
    
    # Copy and shuffle the list
    shuffled_items = list(items)
    rng.shuffle(shuffled_items)

    total = len(shuffled_items)
    if total == 0:
        return [], [], []

    # Calculate split indices
    train_end = int(round(train_ratio * total))
    val_end = train_end + int(round(val_ratio * total))

    # Handle edge case where rounding sums slightly off total
    # Ensure all elements are covered by assigning the rest to test
    train_items = shuffled_items[:train_end]
    val_items = shuffled_items[train_end:val_end]
    test_items = shuffled_items[val_end:]

    logger.info(f"Split dataset: {len(train_items)} train, {len(val_items)} val, {len(test_items)} test.")
    return train_items, val_items, test_items
