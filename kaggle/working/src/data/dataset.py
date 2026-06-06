"""Dataset classes for LLM classification."""

from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from .preprocessor import TextPreprocessor


class PreferenceDataset(Dataset):
    """Dataset for preference prediction task."""
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer: PreTrainedTokenizer,
        preprocessor: TextPreprocessor,
        max_length: int = 3072,
        has_labels: bool = True,
    ):
        """Initialize dataset.
        
        Args:
            data: List of samples with prompt, response_a, response_b
            tokenizer: Tokenizer for encoding
            preprocessor: Text preprocessor for truncation
            max_length: Maximum sequence length
            has_labels: Whether data contains labels
        """
        self.data = data
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.max_length = max_length
        self.has_labels = has_labels
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.
        
        Returns:
            Dictionary with input_ids, attention_mask, and optionally labels
        """
        sample = self.data[idx]
        
        # Preprocess text
        processed = self.preprocessor.preprocess(
            prompt=sample.get("prompt", ""),
            response_a=sample.get("response_a", ""),
            response_b=sample.get("response_b", ""),
        )
        
        # Tokenize
        encoded = self.tokenizer(
            processed["formatted_text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        result = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }
        
        # Add labels if available
        if self.has_labels:
            # Labels: 0 = winner_model_a, 1 = winner_model_b, 2 = winner_tie
            if sample.get("winner_model_a", 0) == 1:
                label = 0
            elif sample.get("winner_model_b", 0) == 1:
                label = 1
            else:
                label = 2
            
            result["labels"] = torch.tensor(label, dtype=torch.long)
        
        return result


class PreferenceCollator:
    """Collator for batching preference samples."""
    
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 3072):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate a batch of samples.
        
        Args:
            batch: List of samples from dataset
            
        Returns:
            Batched tensors
        """
        # Stack tensors
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        
        # Add labels if present in first item
        if "labels" in batch[0]:
            labels = torch.stack([item["labels"] for item in batch])
            result["labels"] = labels
        
        return result


def create_datasets(
    train_data: List[Dict],
    val_data: List[Dict],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 3072,
    **preprocessor_kwargs,
) -> tuple:
    """Create train and validation datasets.
    
    Args:
        train_data: Training samples
        val_data: Validation samples
        tokenizer: Tokenizer
        max_length: Maximum sequence length
        **preprocessor_kwargs: Additional arguments for TextPreprocessor
        
    Returns:
        (train_dataset, val_dataset)
    """
    preprocessor = TextPreprocessor(
        tokenizer=tokenizer,
        max_length=max_length,
        **preprocessor_kwargs,
    )
    
    train_dataset = PreferenceDataset(
        data=train_data,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        max_length=max_length,
        has_labels=True,
    )
    
    val_dataset = PreferenceDataset(
        data=val_data,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        max_length=max_length,
        has_labels=True,
    )
    
    return train_dataset, val_dataset
