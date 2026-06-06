"""Global configuration for LLM classification fine-tuning."""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    model_name: str = "google/gemma-2-9b-it"
    model_path: str = "../../../input/models/emiz6413/gemma-2/transformers/gemma-2-9b-it-4bit/1/gemma-2-9b-it-4bit"
    num_labels: int = 3
    max_length: int = 3072
    
    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # 4-bit quantization
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"


@dataclass
class DataConfig:
    """Data processing configuration."""
    train_path: str = "../../../input/competitions/llm-classification-finetuning/train.csv"
    test_path: str = "../../../input/competitions/llm-classification-finetuning/test.csv"
    sample_submission_path: str = "../../../input/competitions/llm-classification-finetuning/sample_submission.csv"
    
    # Token budget allocation (max_length = 3072)
    template_tokens: int = 150  # Reserved for template text
    prompt_max_tokens: int = 512
    response_max_tokens: int = 1200  # Per response
    
    # Truncation strategy
    prompt_head_ratio: float = 0.4
    prompt_tail_ratio: float = 0.6
    response_head_ratio: float = 0.65
    response_tail_ratio: float = 0.35
    
    # Data augmentation
    use_augmentation: bool = True
    augmentation_prob: float = 0.5
    
    # Stratified split
    validation_ratio: float = 0.1
    random_seed: int = 42


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    output_dir: str = "../../../../outputs/training"
    logging_dir: str = "../../../../outputs/logs"
    
    num_epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    # Optimization
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    
    # Logging and saving
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    save_total_limit: int = 2
    
    # Evaluation
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    load_best_model_at_end: bool = True
    
    # Mixed precision
    fp16: bool = True
    
    # Wandb
    use_wandb: bool = True
    wandb_project: str = "llm-classification"
    wandb_run_name: Optional[str] = None


@dataclass
class InferenceConfig:
    """Inference configuration."""
    output_path: str = "../../../outputs/submission.csv"
    batch_size: int = 4
    
    # Bidirectional prediction for debiasing
    use_bidirectional: bool = True
    
    # TTA (Test Time Augmentation)
    use_tta: bool = False


@dataclass
class Config:
    """Main configuration container."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    def __post_init__(self):
        """Validate and adjust paths."""
        base_dir = Path(__file__).parent.parent.parent
        
        # Convert relative paths to absolute
        self.model.model_path = str(base_dir / self.model.model_path.replace("../../../", ""))
        self.data.train_path = str(base_dir / self.data.train_path.replace("../../../", ""))
        self.data.test_path = str(base_dir / self.data.test_path.replace("../../../", ""))
        self.data.sample_submission_path = str(base_dir / self.data.sample_submission_path.replace("../../../", ""))
        self.training.output_dir = str(base_dir / self.training.output_dir.replace("../../../", ""))
        self.training.logging_dir = str(base_dir / self.training.logging_dir.replace("../../../", ""))
        self.inference.output_path = str(base_dir / self.inference.output_path.replace("../../../", ""))
        
        # Create directories if they don't exist
        Path(self.training.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.training.logging_dir).mkdir(parents=True, exist_ok=True)


def get_config() -> Config:
    """Get default configuration."""
    return Config()
