"""Evaluation metrics for LLM classification.

Primary metric: Log Loss (as per competition requirements).
Secondary metric: Accuracy (for reference).
"""

import numpy as np
from typing import Dict, List
from sklearn.metrics import log_loss, accuracy_score


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute evaluation metrics for HuggingFace Trainer.
    
    Args:
        eval_pred: Tuple of (predictions, labels) from Trainer
        
    Returns:
        Dictionary with metrics
    """
    predictions, labels = eval_pred
    
    # Convert logits to probabilities
    probs = softmax(predictions)
    
    # Ensure labels are integers
    labels = labels.astype(int)
    
    # Compute metrics
    metrics = {
        "log_loss": compute_log_loss(labels, probs),
        "accuracy": compute_accuracy(labels, probs),
    }
    
    return metrics


def compute_log_loss(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    """Compute log loss with epsilon for numerical stability.
    
    Args:
        y_true: True labels (integers 0, 1, 2)
        y_pred: Predicted probabilities (n_samples, 3)
        eps: Epsilon for clipping probabilities
        
    Returns:
        Log loss value
    """
    # Clip probabilities to avoid log(0)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    # Normalize to ensure sum to 1
    y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)
    
    return log_loss(y_true, y_pred)


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute accuracy.
    
    Args:
        y_true: True labels (integers 0, 1, 2)
        y_pred: Predicted probabilities (n_samples, 3)
        
    Returns:
        Accuracy score
    """
    y_pred_classes = np.argmax(y_pred, axis=1)
    return accuracy_score(y_true, y_pred_classes)


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values for each set of scores.
    
    Args:
        x: Input logits (n_samples, n_classes)
        
    Returns:
        Softmax probabilities
    """
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / exp_x.sum(axis=1, keepdims=True)


def compute_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute per-class accuracy metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        
    Returns:
        Dictionary with per-class metrics
    """
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    metrics = {}
    class_names = ["winner_model_a", "winner_model_b", "winner_tie"]
    
    for i, name in enumerate(class_names):
        mask = y_true == i
        if mask.sum() > 0:
            class_acc = (y_pred_classes[mask] == i).mean()
            metrics[f"accuracy_{name}"] = class_acc
    
    return metrics


class MetricsComputer:
    """Stateful metrics computer for tracking during training."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset accumulated predictions."""
        self.all_predictions = []
        self.all_labels = []
    
    def add_batch(self, predictions: np.ndarray, labels: np.ndarray):
        """Add a batch of predictions.
        
        Args:
            predictions: Model predictions (logits or probs)
            labels: True labels
        """
        # Convert logits to probs if needed
        if predictions.ndim == 2 and predictions.shape[1] == 3:
            if not np.allclose(predictions.sum(axis=1), 1.0, atol=0.1):
                predictions = softmax(predictions)
        
        self.all_predictions.append(predictions)
        self.all_labels.append(labels)
    
    def compute(self) -> Dict[str, float]:
        """Compute metrics over all accumulated batches.
        
        Returns:
            Dictionary with metrics
        """
        if not self.all_predictions:
            return {}
        
        y_pred = np.concatenate(self.all_predictions, axis=0)
        y_true = np.concatenate(self.all_labels, axis=0)
        
        metrics = {
            "log_loss": compute_log_loss(y_true, y_pred),
            "accuracy": compute_accuracy(y_true, y_pred),
        }
        
        # Add per-class metrics
        per_class = compute_per_class_metrics(y_true, y_pred)
        metrics.update(per_class)
        
        return metrics
