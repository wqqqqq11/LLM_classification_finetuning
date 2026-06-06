"""Data augmentation for preference classification.

Implements A/B swap augmentation from plan_v1.md:
- Original: (prompt, response_a, response_b) -> label=A
- Swapped: (prompt, response_b, response_a) -> label=B
- Tie labels remain unchanged
"""

import random
from typing import Dict, List


def swap_augmentation(
    data: List[Dict],
    swap_prob: float = 0.5,
    seed: int = 42,
) -> List[Dict]:
    """Apply A/B swap augmentation to training data.
    
    IMPORTANT: Must be applied AFTER train/val split to prevent
    data leakage (original and swapped samples ending up in different splits).
    
    Label mapping:
    - A wins -> B wins (swapped position)
    - B wins -> A wins (swapped position)
    - Tie -> Tie (unchanged)
    
    Args:
        data: Original training samples
        swap_prob: Probability of swapping each sample
        seed: Random seed for reproducibility
        
    Returns:
        Augmented dataset with swapped samples
    """
    random.seed(seed)
    augmented = []
    
    for sample in data:
        # Always keep original
        augmented.append(sample.copy())
        
        # Randomly add swapped version
        if random.random() < swap_prob:
            swapped = swap_single_sample(sample)
            augmented.append(swapped)
    
    return augmented


def swap_single_sample(sample: Dict) -> Dict:
    """Swap response_a and response_b in a single sample.
    
    Args:
        sample: Original sample with prompt, response_a, response_b, and labels
        
    Returns:
        Swapped sample with updated labels
    """
    swapped = sample.copy()
    
    # Swap responses
    swapped["response_a"] = sample["response_b"]
    swapped["response_b"] = sample["response_a"]
    
    # Swap labels (A wins <-> B wins, Tie stays the same)
    orig_winner_a = sample.get("winner_model_a", 0)
    orig_winner_b = sample.get("winner_model_b", 0)
    orig_winner_tie = sample.get("winner_tie", 0)
    
    swapped["winner_model_a"] = orig_winner_b
    swapped["winner_model_b"] = orig_winner_a
    swapped["winner_tie"] = orig_winner_tie
    
    return swapped


def augment_dataset(
    train_data: List[Dict],
    use_augmentation: bool = True,
    swap_prob: float = 0.5,
    seed: int = 42,
) -> List[Dict]:
    """Apply augmentation to training data.
    
    Args:
        train_data: Training samples
        use_augmentation: Whether to apply augmentation
        swap_prob: Probability of swapping each sample
        seed: Random seed
        
    Returns:
        Augmented dataset
    """
    if not use_augmentation:
        return train_data
    
    return swap_augmentation(train_data, swap_prob=swap_prob, seed=seed)
