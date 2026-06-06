"""Main entry point for LLM classification fine-tuning.

Usage:
    Train:   python main.py --mode train
    Predict: python main.py --mode predict --model_path outputs/training/checkpoint-xxx
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "kaggle" / "working" / "src"))

from kaggle.working.src.config import get_config, Config
from kaggle.working.src.data import load_and_split, augment_dataset, create_datasets
from kaggle.working.src.models import create_model
from kaggle.working.src.training import create_training_arguments, create_trainer, compute_metrics
from kaggle.working.src.inference import predict_and_save
from kaggle.working.src.utils import setup_logger
from kaggle.working.src.data.preprocessor import TextPreprocessor


def train(config: Config):
    """Run training pipeline.
    
    Args:
        config: Configuration object
    """
    # Setup logging
    logger = setup_logger(
        name="train",
        log_dir=config.training.logging_dir,
    )
    logger.info("Starting training pipeline")
    
    # Check GPU availability
    import torch
    if torch.cuda.is_available():
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("GPU not available, training on CPU (will be very slow)")
    
    # Load and split data
    logger.info("Loading and splitting data")
    train_data, val_data, test_data = load_and_split(
        train_path=config.data.train_path,
        test_path=config.data.test_path,
        validation_ratio=config.data.validation_ratio,
        random_state=config.data.random_seed,
    )
    
    # Apply augmentation to training data only (AFTER split)
    if config.data.use_augmentation:
        logger.info("Applying data augmentation to training set")
        train_data = augment_dataset(
            train_data,
            use_augmentation=True,
            swap_prob=config.data.augmentation_prob,
            seed=config.data.random_seed,
        )
        logger.info(f"Training set size after augmentation: {len(train_data)}")
    
    # Create model and tokenizer
    logger.info("Creating model")
    model, tokenizer = create_model(
        model_path=config.model.model_path,
        num_labels=config.model.num_labels,
        max_length=config.model.max_length,
        lora_r=config.model.lora_r,
        lora_alpha=config.model.lora_alpha,
        lora_dropout=config.model.lora_dropout,
        lora_target_modules=config.model.lora_target_modules,
        load_in_4bit=config.model.load_in_4bit,
    )
    
    # Verify model device
    device = next(model.parameters()).device
    logger.info(f"Model loaded on device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    
    # Create preprocessor
    preprocessor = TextPreprocessor(
        tokenizer=tokenizer,
        max_length=config.model.max_length,
        template_tokens=config.data.template_tokens,
        prompt_max_tokens=config.data.prompt_max_tokens,
        response_max_tokens=config.data.response_max_tokens,
        prompt_head_ratio=config.data.prompt_head_ratio,
        prompt_tail_ratio=config.data.prompt_tail_ratio,
        response_head_ratio=config.data.response_head_ratio,
        response_tail_ratio=config.data.response_tail_ratio,
    )
    
    # Create datasets
    logger.info("Creating datasets")
    train_dataset, val_dataset = create_datasets(
        train_data=train_data,
        val_data=val_data,
        tokenizer=tokenizer,
        max_length=config.model.max_length,
        template_tokens=config.data.template_tokens,
        prompt_max_tokens=config.data.prompt_max_tokens,
        response_max_tokens=config.data.response_max_tokens,
        prompt_head_ratio=config.data.prompt_head_ratio,
        prompt_tail_ratio=config.data.prompt_tail_ratio,
        response_head_ratio=config.data.response_head_ratio,
        response_tail_ratio=config.data.response_tail_ratio,
    )
    
    # Create training arguments
    logger.info("Setting up training")
    training_args = create_training_arguments(
        output_dir=config.training.output_dir,
        logging_dir=config.training.logging_dir,
        num_epochs=config.training.num_epochs,
        batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        max_grad_norm=config.training.max_grad_norm,
        logging_steps=config.training.logging_steps,
        eval_steps=config.training.eval_steps,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        load_best_model_at_end=config.training.load_best_model_at_end,
        fp16=config.training.fp16,
        use_wandb=config.training.use_wandb,
        wandb_project=config.training.wandb_project,
        wandb_run_name=config.training.wandb_run_name,
    )
    
    # Create trainer
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        training_args=training_args,
        use_early_stopping=True,
        early_stopping_patience=3,
    )
    
    # Train
    logger.info("Starting training")
    trainer.train()
    
    # Save final model
    final_output_dir = Path(config.training.output_dir) / "final"
    logger.info(f"Saving final model to {final_output_dir}")
    trainer.save_model(str(final_output_dir))
    
    # Evaluate on validation set
    logger.info("Evaluating on validation set")
    eval_results = trainer.evaluate()
    logger.info(f"Validation results: {eval_results}")
    
    logger.info("Training complete")
    
    return trainer


def predict(config: Config, model_path: str):
    """Run prediction pipeline.
    
    Args:
        config: Configuration object
        model_path: Path to trained model checkpoint
    """
    # Setup logging
    logger = setup_logger(
        name="predict",
        log_dir=config.training.logging_dir,
    )
    logger.info("Starting prediction pipeline")
    
    # Load data
    logger.info("Loading test data")
    _, _, test_data = load_and_split(
        train_path=config.data.train_path,
        test_path=config.data.test_path,
        validation_ratio=config.data.validation_ratio,
        random_state=config.data.random_seed,
    )
    
    # Load model and tokenizer
    logger.info(f"Loading model from {model_path}")
    model, tokenizer = create_model(
        model_path=model_path,
        num_labels=config.model.num_labels,
        max_length=config.model.max_length,
        load_in_4bit=config.model.load_in_4bit,
    )
    
    # Create preprocessor
    preprocessor = TextPreprocessor(
        tokenizer=tokenizer,
        max_length=config.model.max_length,
        template_tokens=config.data.template_tokens,
        prompt_max_tokens=config.data.prompt_max_tokens,
        response_max_tokens=config.data.response_max_tokens,
        prompt_head_ratio=config.data.prompt_head_ratio,
        prompt_tail_ratio=config.data.prompt_tail_ratio,
        response_head_ratio=config.data.response_head_ratio,
        response_tail_ratio=config.data.response_tail_ratio,
    )
    
    # Run prediction
    logger.info("Running prediction")
    submission = predict_and_save(
        model=model,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        test_data=test_data,
        output_path=config.inference.output_path,
        use_bidirectional=config.inference.use_bidirectional,
        batch_size=config.inference.batch_size,
        max_length=config.model.max_length,
    )
    
    logger.info(f"Submission saved to {config.inference.output_path}")
    logger.info(f"Submission shape: {submission.shape}")
    logger.info(f"Submission preview:\n{submission.head()}")
    
    return submission


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LLM Classification Fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    Train:   python main.py --mode train
    Predict: python main.py --mode predict --model_path outputs/training/final
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "predict"],
        required=True,
        help="Mode: train or predict"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model checkpoint (required for predict mode)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_config()
    
    if args.mode == "train":
        train(config)
    elif args.mode == "predict":
        if args.model_path is None:
            parser.error("--model_path is required for predict mode")
        predict(config, args.model_path)
    else:
        parser.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
