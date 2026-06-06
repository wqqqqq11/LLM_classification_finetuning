"""Training setup for LLM classification using HuggingFace Trainer."""

import os
from pathlib import Path
from typing import Optional

from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from transformers.integrations import WandbCallback

from .evaluation import compute_metrics


def create_training_arguments(
    output_dir: str,
    logging_dir: str,
    num_epochs: int = 3,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    max_grad_norm: float = 1.0,
    logging_steps: int = 10,
    eval_steps: int = 100,
    save_steps: int = 500,
    save_total_limit: int = 2,
    load_best_model_at_end: bool = True,
    fp16: bool = True,
    use_wandb: bool = True,
    wandb_project: str = "llm-classification",
    wandb_run_name: Optional[str] = None,
) -> TrainingArguments:
    """Create TrainingArguments for HuggingFace Trainer.
    
    Args:
        output_dir: Directory to save model checkpoints
        logging_dir: Directory for logs
        num_epochs: Number of training epochs
        batch_size: Per-device batch size
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Learning rate
        weight_decay: Weight decay
        warmup_ratio: Warmup ratio
        max_grad_norm: Max gradient norm for clipping
        logging_steps: Log every N steps
        eval_steps: Evaluate every N steps
        save_steps: Save checkpoint every N steps
        save_total_limit: Maximum checkpoints to keep
        load_best_model_at_end: Load best model at end
        fp16: Use mixed precision training
        use_wandb: Use Weights & Biases logging
        wandb_project: Wandb project name
        wandb_run_name: Wandb run name
        
    Returns:
        TrainingArguments instance
    """
    # Setup wandb integration
    report_to = ["wandb"] if use_wandb else []
    
    if use_wandb:
        os.environ["WANDB_PROJECT"] = wandb_project
        if wandb_run_name:
            os.environ["WANDB_NAME"] = wandb_run_name
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        
        # Training hyperparameters
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        max_grad_norm=max_grad_norm,
        lr_scheduler_type="cosine",
        
        # Logging and evaluation
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        
        # Best model selection
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Mixed precision
        fp16=fp16,
        
        # Reporting
        report_to=report_to,
        logging_first_step=True,
        
        # Remove unused columns (we handle this in dataset)
        remove_unused_columns=False,
        
        # Windows compatibility - disable multiprocessing and pin_memory
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        
        # Seed for reproducibility
        seed=42,
    )
    
    return training_args


def create_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    training_args: TrainingArguments,
    use_early_stopping: bool = True,
    early_stopping_patience: int = 3,
) -> Trainer:
    """Create HuggingFace Trainer instance.
    
    Args:
        model: Model to train
        tokenizer: Tokenizer
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        training_args: Training arguments
        use_early_stopping: Whether to use early stopping
        early_stopping_patience: Patience for early stopping
        
    Returns:
        Trainer instance
    """
    # Setup callbacks
    callbacks = []
    
    if use_early_stopping:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
            )
        )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    
    return trainer


def train_model(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir: str,
    logging_dir: str,
    num_epochs: int = 3,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    use_wandb: bool = True,
    wandb_project: str = "llm-classification",
    wandb_run_name: Optional[str] = None,
) -> Trainer:
    """Train the model with specified configuration.
    
    Args:
        model: Model to train
        tokenizer: Tokenizer
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        output_dir: Output directory for checkpoints
        logging_dir: Logging directory
        num_epochs: Number of epochs
        batch_size: Batch size
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Learning rate
        use_wandb: Use Wandb logging
        wandb_project: Wandb project name
        wandb_run_name: Wandb run name
        
    Returns:
        Trained Trainer instance
    """
    # Create training arguments
    training_args = create_training_arguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
    )
    
    # Create trainer
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_args=training_args,
    )
    
    # Train
    trainer.train()
    
    return trainer
