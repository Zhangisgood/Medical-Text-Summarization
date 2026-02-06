"""
Training Script for Medical Text Summarization

This script fine-tunes FLAN-T5 using LoRA (Low-Rank Adaptation) for
parameter-efficient fine-tuning on medical dialogue summarization.

Usage:
    python train.py --config config1
    python train.py --learning_rate 5e-5 --batch_size 8 --num_epochs 3
"""

import os
import argparse
import json
from pathlib import Path
from datetime import datetime
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import load_from_disk
import numpy as np
import evaluate


# Predefined hyperparameter configurations
CONFIGS = {
    "config1": {
        "learning_rate": 2e-5,
        "batch_size": 8,
        "num_epochs": 3,
        "lora_rank": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "weight_decay": 0.01,
    },
    "config2": {
        "learning_rate": 5e-5,
        "batch_size": 16,
        "num_epochs": 5,
        "lora_rank": 16,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "weight_decay": 0.001,
    },
    "config3": {
        "learning_rate": 1e-5,
        "batch_size": 4,
        "num_epochs": 3,
        "lora_rank": 4,
        "lora_alpha": 16,
        "lora_dropout": 0.15,
        "weight_decay": 0.05,
    },
}


class MedicalSummarizationTrainer:
    """Handles model training with LoRA for medical summarization."""
    
    def __init__(self, args):
        """Initialize trainer with arguments."""
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Set up output directories
        self.setup_directories()
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.model = self.load_model_with_lora()
        
        # Load metrics
        self.rouge_metric = evaluate.load("rouge")
        
    def setup_directories(self):
        """Create necessary directories for outputs."""
        self.output_dir = Path(self.args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(self.args.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def load_model_with_lora(self):
        """Load base model and apply LoRA configuration."""
        print(f"\nLoading base model: {self.args.model_name}")
        
        # Load base model
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.args.model_name,
            torch_dtype=torch.float16 if self.args.use_fp16 else torch.float32,
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=self.args.lora_rank,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            target_modules=["q", "v"],  # For T5 models
            inference_mode=False,
        )
        
        # Apply LoRA to model
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        model.print_trainable_parameters()
        
        return model
    
    def load_datasets(self):
        """Load preprocessed datasets."""
        print(f"\nLoading datasets from: {self.args.data_dir}")
        
        dataset_dict = load_from_disk(self.args.data_dir)
        
        print(f"Train samples: {len(dataset_dict['train'])}")
        print(f"Validation samples: {len(dataset_dict['validation'])}")
        print(f"Test samples: {len(dataset_dict['test'])}")
        
        return dataset_dict
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        
        # Decode predictions and labels
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Replace -100 in labels as we can't decode them
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Compute ROUGE scores
        result = self.rouge_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )
        
        # Extract and round scores
        result = {
            "rouge1": round(result["rouge1"], 4),
            "rouge2": round(result["rouge2"], 4),
            "rougeL": round(result["rougeL"], 4),
        }
        
        return result
    
    def train(self):
        """Execute training process."""
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        
        # Load datasets
        dataset_dict = self.load_datasets()
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
        )
        
        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(self.output_dir),
            
            # Training hyperparameters
            num_train_epochs=self.args.num_epochs,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.batch_size,
            learning_rate=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            
            # Optimization
            fp16=self.args.use_fp16 and torch.cuda.is_available(),
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            warmup_steps=self.args.warmup_steps,
            max_grad_norm=1.0,
            
            # Evaluation and saving
            evaluation_strategy="steps",
            eval_steps=self.args.eval_steps,
            save_strategy="steps",
            save_steps=self.args.save_steps,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="rouge1",
            greater_is_better=True,
            
            # Logging
            logging_dir=str(self.log_dir),
            logging_steps=self.args.logging_steps,
            report_to=["tensorboard"],
            
            # Generation (for evaluation)
            predict_with_generate=True,
            generation_max_length=150,
            generation_num_beams=4,
            
            # Other
            seed=42,
            remove_unused_columns=True,
        )
        
        # Initialize trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset_dict["train"],
            eval_dataset=dataset_dict["validation"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )
        
        # Train
        print("\nStarting training...")
        train_result = trainer.train()
        
        # Save final model
        print("\nSaving final model...")
        trainer.save_model(str(self.output_dir / "final_model"))
        self.tokenizer.save_pretrained(str(self.output_dir / "final_model"))
        
        # Save training metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        # Evaluate on validation set
        print("\nEvaluating on validation set...")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
        
        # Save training configuration
        config_path = self.output_dir / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(vars(self.args), f, indent=2, default=str)
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        print(f"\nModel saved to: {self.output_dir / 'final_model'}")
        print(f"Logs saved to: {self.log_dir}")
        print("\nValidation Metrics:")
        for key, value in eval_metrics.items():
            if key.startswith("eval_"):
                print(f"  {key}: {value:.4f}")
        
        return trainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train medical summarization model")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="google/flan-t5-base",
                       help="Pretrained model name")
    parser.add_argument("--data_dir", type=str, default="data/processed/dataset_dict",
                       help="Directory containing processed datasets")
    parser.add_argument("--output_dir", type=str, default="models/finetuned",
                       help="Output directory for model checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs",
                       help="Directory for training logs")
    
    # Configuration preset
    parser.add_argument("--config", type=str, choices=["config1", "config2", "config3"],
                       help="Use predefined configuration")
    
    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    
    # LoRA parameters
    parser.add_argument("--lora_rank", type=int, default=8,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                       help="LoRA dropout")
    
    # Optimization
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                       help="Gradient accumulation steps")
    parser.add_argument("--warmup_steps", type=int, default=500,
                       help="Warmup steps")
    parser.add_argument("--use_fp16", action="store_true", default=True,
                       help="Use mixed precision training")
    
    # Logging and evaluation
    parser.add_argument("--logging_steps", type=int, default=50,
                       help="Logging frequency")
    parser.add_argument("--eval_steps", type=int, default=200,
                       help="Evaluation frequency")
    parser.add_argument("--save_steps", type=int, default=200,
                       help="Save checkpoint frequency")
    
    args = parser.parse_args()
    
    # Apply config preset if specified
    if args.config:
        config = CONFIGS[args.config]
        for key, value in config.items():
            setattr(args, key, value)
        print(f"\nUsing configuration: {args.config}")
        print(json.dumps(config, indent=2))
    
    return args


def main():
    """Main execution function."""
    print("="*60)
    print("Medical Text Summarization - Model Training")
    print("="*60)
    
    # Parse arguments
    args = parse_args()
    
    # Print configuration
    print("\nTraining Configuration:")
    print(f"  Model: {args.model_name}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  LoRA Rank: {args.lora_rank}")
    print(f"  LoRA Alpha: {args.lora_alpha}")
    
    # Initialize trainer
    trainer = MedicalSummarizationTrainer(args)
    
    # Train model
    trainer.train()
    
    print("\nNext steps:")
    print("  1. View training logs: tensorboard --logdir logs/")
    print("  2. Evaluate model: python src/evaluate.py")
    print("  3. Run inference: python src/inference.py")


if __name__ == "__main__":
    main()
