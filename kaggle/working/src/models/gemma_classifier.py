"""Gemma 2 9B IT with LoRA for preference classification."""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training,
)


class GemmaPreferenceClassifier:
    """Gemma 2 9B IT with LoRA for 3-class preference classification."""
    
    def __init__(
        self,
        model_path: str,
        num_labels: int = 3,
        max_length: int = 3072,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[list] = None,
        load_in_4bit: bool = True,
        device_map: str = "auto",
    ):
        """Initialize the classifier.
        
        Args:
            model_path: Path to Gemma model
            num_labels: Number of classification labels (3 for A/B/Tie)
            max_length: Maximum sequence length
            lora_r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            lora_target_modules: Modules to apply LoRA
            load_in_4bit: Whether to load in 4-bit quantization
            device_map: Device mapping strategy
        """
        self.model_path = model_path
        self.num_labels = num_labels
        self.max_length = max_length
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map
        
        # Default target modules for Gemma
        if lora_target_modules is None:
            lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]
        self.lora_target_modules = lora_target_modules
        
        self.model = None
        self.tokenizer = None
    
    def load_tokenizer(self) -> PreTrainedTokenizer:
        """Load the tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        self.tokenizer = tokenizer
        return tokenizer
    
    def create_quantization_config(self) -> BitsAndBytesConfig:
        """Create 4-bit quantization configuration."""
        return BitsAndBytesConfig(
            load_in_4bit=self.load_in_4bit,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    
    def create_lora_config(self) -> LoraConfig:
        """Create LoRA configuration."""
        return LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.lora_target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
            inference_mode=False,
        )
    
    def load_base_model(self) -> AutoModelForSequenceClassification:
        """Load the base model with optional quantization."""
        quantization_config = None
        if self.load_in_4bit:
            quantization_config = self.create_quantization_config()
        
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            num_labels=self.num_labels,
            quantization_config=quantization_config,
            device_map=self.device_map,
            trust_remote_code=True,
            torch_dtype=torch.float16 if not self.load_in_4bit else None,
        )
        
        # Prepare for k-bit training if using quantization
        if self.load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        
        return model
    
    def setup(self) -> Tuple[AutoModelForSequenceClassification, PreTrainedTokenizer]:
        """Setup the model and tokenizer with LoRA.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        import torch
        
        # Load tokenizer first
        self.load_tokenizer()
        
        # Load base model
        self.model = self.load_base_model()
        
        # Verify model is on GPU if available
        if torch.cuda.is_available():
            device = next(self.model.parameters()).device
            if device.type != "cuda":
                print(f"Warning: Model on {device}, moving to CUDA...")
                self.model = self.model.to("cuda")
        
        # Apply LoRA
        lora_config = self.create_lora_config()
        self.model = get_peft_model(self.model, lora_config)
        
        # Set pad token id in model config
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        return self.model, self.tokenizer
    
    def save(self, output_dir: str):
        """Save the model and tokenizer.
        
        Args:
            output_dir: Directory to save model
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA adapter
        self.model.save_pretrained(output_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
    
    def load_adapter(self, adapter_path: str):
        """Load LoRA adapter weights.
        
        Args:
            adapter_path: Path to adapter weights
        """
        if self.model is None:
            raise ValueError("Model must be setup before loading adapter")
        
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
    
    def merge_and_unload(self):
        """Merge LoRA weights into base model for inference."""
        if isinstance(self.model, PeftModel):
            self.model = self.model.merge_and_unload()
    
    def get_trainable_parameters(self) -> Tuple[int, int, float]:
        """Get statistics about trainable parameters.
        
        Returns:
            (trainable_params, all_params, percentage)
        """
        trainable_params = 0
        all_params = 0
        
        for _, param in self.model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        percentage = 100 * trainable_params / all_params if all_params > 0 else 0
        
        return trainable_params, all_params, percentage
    
    def print_trainable_parameters(self):
        """Print trainable parameter statistics."""
        trainable, total, pct = self.get_trainable_parameters()
        print(f"Trainable params: {trainable:,} || "
              f"All params: {total:,} || "
              f"Trainable %: {pct:.4f}%")


def create_model(
    model_path: str,
    num_labels: int = 3,
    max_length: int = 3072,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[list] = None,
    load_in_4bit: bool = True,
) -> Tuple[AutoModelForSequenceClassification, PreTrainedTokenizer]:
    """Create and setup the Gemma preference classifier.
    
    Args:
        model_path: Path to model
        num_labels: Number of labels
        max_length: Max sequence length
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        lora_target_modules: Target modules for LoRA
        load_in_4bit: Use 4-bit quantization
        
    Returns:
        Tuple of (model, tokenizer)
    """
    classifier = GemmaPreferenceClassifier(
        model_path=model_path,
        num_labels=num_labels,
        max_length=max_length,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        load_in_4bit=load_in_4bit,
    )
    
    model, tokenizer = classifier.setup()
    classifier.print_trainable_parameters()
    
    return model, tokenizer
