"""Inference predictor for LLM classification with bidirectional debiasing.

Implements bidirectional prediction from plan_v1.md:
1. First pass: (prompt, response_a, response_b) -> probs
2. Second pass: (prompt, response_b, response_a) -> probs
3. Average after mapping labels back
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from tqdm import tqdm


class PreferencePredictor:
    """Predictor for preference classification with bidirectional debiasing."""
    
    def __init__(
        self,
        model,
        tokenizer,
        preprocessor,
        max_length: int = 3072,
        batch_size: int = 4,
        use_bidirectional: bool = True,
        device: str = "cuda",
    ):
        """Initialize predictor.
        
        Args:
            model: Trained model
            tokenizer: Tokenizer
            preprocessor: Text preprocessor
            max_length: Maximum sequence length
            batch_size: Inference batch size
            use_bidirectional: Use bidirectional prediction for debiasing
            device: Device to run inference on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_bidirectional = use_bidirectional
        self.device = device
        
        # Set model to eval mode
        self.model.eval()
        self.model.to(device)
    
    def predict_single(
        self,
        prompt: str,
        response_a: str,
        response_b: str,
    ) -> np.ndarray:
        """Make prediction for a single sample.
        
        Args:
            prompt: User prompt
            response_a: Response A
            response_b: Response B
            
        Returns:
            Probabilities for (winner_a, winner_b, tie)
        """
        # Preprocess
        processed = self.preprocessor.preprocess(prompt, response_a, response_b)
        
        # Tokenize
        encoded = self.tokenizer(
            processed["formatted_text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Move to device
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        
        return probs.cpu().numpy()[0]
    
    def predict_bidirectional(
        self,
        prompt: str,
        response_a: str,
        response_b: str,
    ) -> np.ndarray:
        """Make bidirectional prediction for debiasing.
        
        First pass: (prompt, response_a, response_b) -> (p_a, p_b, p_tie)
        Second pass: (prompt, response_b, response_a) -> (p'_a, p'_b, p'_tie)
        Map back: swapped_p_a = p'_b, swapped_p_b = p'_a, swapped_p_tie = p'_tie
        Average: final = (first + swapped) / 2
        
        Args:
            prompt: User prompt
            response_a: Response A
            response_b: Response B
            
        Returns:
            Averaged probabilities for (winner_a, winner_b, tie)
        """
        # First pass
        probs_first = self.predict_single(prompt, response_a, response_b)
        
        if not self.use_bidirectional:
            return probs_first
        
        # Second pass with swapped responses
        probs_swapped = self.predict_single(prompt, response_b, response_a)
        
        # Map labels back: in swapped prediction, what was labeled as A is actually B
        # probs_swapped = [p'_a, p'_b, p'_tie] where A/B are swapped
        # So: swapped_p_a = p'_b, swapped_p_b = p'_a, swapped_p_tie = p'_tie
        mapped_swapped = np.array([
            probs_swapped[1],  # p'_b is actually probability for original A
            probs_swapped[0],  # p'_a is actually probability for original B
            probs_swapped[2],  # p'_tie is unchanged
        ])
        
        # Average the two predictions
        final_probs = (probs_first + mapped_swapped) / 2.0
        
        # Normalize to ensure sum to 1
        final_probs = final_probs / final_probs.sum()
        
        return final_probs
    
    def predict_batch(
        self,
        data: List[Dict],
        show_progress: bool = True,
    ) -> np.ndarray:
        """Predict for a batch of samples.
        
        Args:
            data: List of samples with prompt, response_a, response_b
            show_progress: Show progress bar
            
        Returns:
            Array of probabilities (n_samples, 3)
        """
        all_probs = []
        
        iterator = tqdm(data) if show_progress else data
        
        for sample in iterator:
            prompt = sample.get("prompt", "")
            response_a = sample.get("response_a", "")
            response_b = sample.get("response_b", "")
            
            if self.use_bidirectional:
                probs = self.predict_bidirectional(prompt, response_a, response_b)
            else:
                probs = self.predict_single(prompt, response_a, response_b)
            
            all_probs.append(probs)
        
        return np.array(all_probs)
    
    def create_submission(
        self,
        data: List[Dict],
        output_path: str,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Create submission file.
        
        Args:
            data: Test data with id, prompt, response_a, response_b
            output_path: Path to save submission CSV
            show_progress: Show progress bar
            
        Returns:
            Submission DataFrame
        """
        # Get predictions
        probs = self.predict_batch(data, show_progress=show_progress)
        
        # Extract IDs
        ids = [sample.get("id", i) for i, sample in enumerate(data)]
        
        # Create DataFrame
        submission = pd.DataFrame({
            "id": ids,
            "winner_model_a": probs[:, 0],
            "winner_model_b": probs[:, 1],
            "winner_tie": probs[:, 2],
        })
        
        # Ensure probabilities sum to 1
        prob_sum = submission[["winner_model_a", "winner_model_b", "winner_tie"]].sum(axis=1)
        submission["winner_model_a"] /= prob_sum
        submission["winner_model_b"] /= prob_sum
        submission["winner_tie"] /= prob_sum
        
        # Save to CSV
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(output_path, index=False)
        
        return submission


def predict_and_save(
    model,
    tokenizer,
    preprocessor,
    test_data: List[Dict],
    output_path: str,
    use_bidirectional: bool = True,
    batch_size: int = 4,
    max_length: int = 3072,
) -> pd.DataFrame:
    """Convenience function for prediction and submission generation.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        preprocessor: Text preprocessor
        test_data: Test data
        output_path: Path to save submission
        use_bidirectional: Use bidirectional prediction
        batch_size: Batch size for inference
        max_length: Maximum sequence length
        
    Returns:
        Submission DataFrame
    """
    predictor = PreferencePredictor(
        model=model,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        max_length=max_length,
        batch_size=batch_size,
        use_bidirectional=use_bidirectional,
    )
    
    submission = predictor.create_submission(
        data=test_data,
        output_path=output_path,
    )
    
    return submission
