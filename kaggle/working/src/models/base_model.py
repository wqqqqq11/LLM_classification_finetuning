"""Base model interface for preference classification."""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
import torch
from transformers import PreTrainedTokenizer


class BasePreferenceModel(ABC):
    """Abstract base class for preference classification models."""
    
    def __init__(
        self,
        model_name: str,
        num_labels: int = 3,
        max_length: int = 3072,
    ):
        """Initialize base model.
        
        Args:
            model_name: Model identifier or path
            num_labels: Number of classification labels
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
    
    @abstractmethod
    def setup(self) -> Tuple[torch.nn.Module, PreTrainedTokenizer]:
        """Setup model and tokenizer.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        pass
    
    @abstractmethod
    def save(self, output_dir: str):
        """Save model and tokenizer.
        
        Args:
            output_dir: Directory to save model
        """
        pass
    
    @abstractmethod
    def load(self, model_path: str):
        """Load model from path.
        
        Args:
            model_path: Path to saved model
        """
        pass
    
    @abstractmethod
    def get_trainable_parameters(self) -> Tuple[int, int, float]:
        """Get trainable parameter statistics.
        
        Returns:
            (trainable_params, all_params, percentage)
        """
        pass
    
    def print_trainable_parameters(self):
        """Print trainable parameter statistics."""
        trainable, total, pct = self.get_trainable_parameters()
        print(f"Trainable params: {trainable:,} || "
              f"All params: {total:,} || "
              f"Trainable %: {pct:.4f}%")
    
    def predict(self, texts: list) -> torch.Tensor:
        """Make predictions on input texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            Logits tensor
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not initialized. Call setup() first.")
        
        self.model.eval()
        
        encoded = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        
        device = next(self.model.parameters()).device
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        return outputs.logits
