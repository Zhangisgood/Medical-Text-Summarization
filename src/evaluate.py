"""
Evaluation Script for Medical Text Summarization

This script evaluates the fine-tuned model on the test set and compares
performance with the baseline (pre-trained) model.

Usage:
    python evaluate.py --model_path models/finetuned/final_model
"""

import os
import argparse
import json
from pathlib import Path
from typing import Dict, List
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
from datasets import load_from_disk
import evaluate
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class ModelEvaluator:
    """Evaluates model performance on medical summarization task."""
    
    def __init__(self, model_path: str, baseline_model: str = "google/flan-t5-base"):
        """Initialize evaluator with model paths."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load fine-tuned model
        print(f"\nLoading fine-tuned model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.finetuned_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.finetuned_model.to(self.device)
        self.finetuned_model.eval()
        
        # Load baseline model
        print(f"Loading baseline model: {baseline_model}")
        self.baseline_model = AutoModelForSeq2SeqLM.from_pretrained(baseline_model)
        self.baseline_model.to(self.device)
        self.baseline_model.eval()
        
        # Load metrics
        self.rouge_metric = evaluate.load("rouge")
        self.bertscore_metric = evaluate.load("bertscore")
        
    def generate_summary(self, model, input_text: str, max_length: int = 150) -> str:
        """Generate summary for given input text."""
        inputs = self.tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                length_penalty=0.6,
                early_stopping=True,
            )
        
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
    
    def evaluate_model(self, model, test_dataset, model_name: str) -> Dict:
        """Evaluate model on test dataset."""
        print(f"\nEvaluating {model_name}...")
        
        predictions = []
        references = []
        
        # Generate predictions
        for example in tqdm(test_dataset, desc=f"Generating summaries ({model_name})"):
            # Reconstruct input
            input_text = f"{example['instruction']}\n\nInput: {example['input']}"
            
            # Generate summary
            pred = self.generate_summary(model, input_text)
            predictions.append(pred)
            references.append(example['output'])
        
        # Compute ROUGE scores
        rouge_results = self.rouge_metric.compute(
            predictions=predictions,
            references=references,
            use_stemmer=True,
        )
        
        # Compute BERTScore
        print(f"Computing BERTScore for {model_name}...")
        bertscore_results = self.bertscore_metric.compute(
            predictions=predictions,
            references=references,
            lang="en",
            model_type="distilbert-base-uncased",
        )
        
        # Aggregate results
        results = {
            "rouge1": round(rouge_results["rouge1"], 4),
            "rouge2": round(rouge_results["rouge2"], 4),
            "rougeL": round(rouge_results["rougeL"], 4),
            "bertscore_precision": round(np.mean(bertscore_results["precision"]), 4),
            "bertscore_recall": round(np.mean(bertscore_results["recall"]), 4),
            "bertscore_f1": round(np.mean(bertscore_results["f1"]), 4),
        }
        
        return results, predictions, references
    
    def compare_models(self, test_dataset) -> Dict:
        """Compare fine-tuned model with baseline."""
        print("\n" + "="*60)
        print("Model Comparison")
        print("="*60)
        
        # Evaluate baseline
        baseline_results, baseline_preds, references = self.evaluate_model(
            self.baseline_model, test_dataset, "Baseline"
        )
        
        # Evaluate fine-tuned model
        finetuned_results, finetuned_preds, references = self.evaluate_model(
            self.finetuned_model, test_dataset, "Fine-tuned"
        )
        
        # Calculate improvements
        improvements = {}
        for metric in baseline_results.keys():
            baseline_val = baseline_results[metric]
            finetuned_val = finetuned_results[metric]
            improvement = ((finetuned_val - baseline_val) / baseline_val) * 100
            improvements[metric] = round(improvement, 2)
        
        comparison = {
            "baseline": baseline_results,
            "finetuned": finetuned_results,
            "improvements": improvements,
        }
        
        return comparison, baseline_preds, finetuned_preds, references
    
    def create_visualizations(self, comparison: Dict, output_dir: Path):
        """Create comparison visualizations."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        
        # 1. Bar chart comparing metrics
        metrics = list(comparison["baseline"].keys())
        baseline_values = [comparison["baseline"][m] for m in metrics]
        finetuned_values = [comparison["finetuned"][m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots()
        bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline', alpha=0.8)
        bars2 = ax.bar(x + width/2, finetuned_values, width, label='Fine-tuned', alpha=0.8)
        
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "metrics_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Improvement percentage chart
        improvements = comparison["improvements"]
        metrics = list(improvements.keys())
        values = [improvements[m] for m in metrics]
        
        fig, ax = plt.subplots()
        colors = ['green' if v > 0 else 'red' for v in values]
        bars = ax.barh(metrics, values, color=colors, alpha=0.7)
        
        ax.set_xlabel('Improvement (%)', fontsize=12)
        ax.set_title('Performance Improvement Over Baseline', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (metric, value) in enumerate(zip(metrics, values)):
            ax.text(value + 1, i, f'{value:+.1f}%', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / "improvement_chart.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nVisualizations saved to: {output_dir}")
    
    def save_results(self, comparison: Dict, output_dir: Path):
        """Save evaluation results to file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        results_path = output_dir / "evaluation_results.json"
        with open(results_path, "w") as f:
            json.dump(comparison, f, indent=2)
        
        # Save as formatted text
        report_path = output_dir / "evaluation_report.txt"
        with open(report_path, "w") as f:
            f.write("="*60 + "\n")
            f.write("Medical Text Summarization - Evaluation Report\n")
            f.write("="*60 + "\n\n")
            
            f.write("BASELINE MODEL PERFORMANCE\n")
            f.write("-"*60 + "\n")
            for metric, value in comparison["baseline"].items():
                f.write(f"  {metric:25s}: {value:.4f}\n")
            
            f.write("\nFINE-TUNED MODEL PERFORMANCE\n")
            f.write("-"*60 + "\n")
            for metric, value in comparison["finetuned"].items():
                f.write(f"  {metric:25s}: {value:.4f}\n")
            
            f.write("\nIMPROVEMENT OVER BASELINE\n")
            f.write("-"*60 + "\n")
            for metric, value in comparison["improvements"].items():
                f.write(f"  {metric:25s}: {value:+.2f}%\n")
        
        print(f"\nResults saved to:")
        print(f"  - {results_path}")
        print(f"  - {report_path}")
    
    def save_example_predictions(self, baseline_preds: List[str], 
                                 finetuned_preds: List[str],
                                 references: List[str],
                                 output_dir: Path,
                                 num_examples: int = 10):
        """Save example predictions for qualitative analysis."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        examples_path = output_dir / "example_predictions.txt"
        with open(examples_path, "w") as f:
            f.write("="*80 + "\n")
            f.write("Example Predictions\n")
            f.write("="*80 + "\n\n")
            
            for i in range(min(num_examples, len(references))):
                f.write(f"Example {i+1}\n")
                f.write("-"*80 + "\n")
                f.write(f"Reference Summary:\n{references[i]}\n\n")
                f.write(f"Baseline Prediction:\n{baseline_preds[i]}\n\n")
                f.write(f"Fine-tuned Prediction:\n{finetuned_preds[i]}\n\n")
                f.write("="*80 + "\n\n")
        
        print(f"Example predictions saved to: {examples_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Evaluate medical summarization model")
    parser.add_argument("--model_path", type=str, default="models/finetuned/final_model",
                       help="Path to fine-tuned model")
    parser.add_argument("--baseline_model", type=str, default="google/flan-t5-base",
                       help="Baseline model name")
    parser.add_argument("--data_dir", type=str, default="data/processed/dataset_dict",
                       help="Directory containing processed datasets")
    parser.add_argument("--output_dir", type=str, default="results/metrics",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Medical Text Summarization - Model Evaluation")
    print("="*60)
    
    # Load test dataset
    print(f"\nLoading test dataset from: {args.data_dir}")
    dataset_dict = load_from_disk(args.data_dir)
    test_dataset = dataset_dict["test"]
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.model_path, args.baseline_model)
    
    # Compare models
    comparison, baseline_preds, finetuned_preds, references = evaluator.compare_models(test_dataset)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print("\nBaseline Model:")
    for metric, value in comparison["baseline"].items():
        print(f"  {metric:25s}: {value:.4f}")
    
    print("\nFine-tuned Model:")
    for metric, value in comparison["finetuned"].items():
        print(f"  {metric:25s}: {value:.4f}")
    
    print("\nImprovement:")
    for metric, value in comparison["improvements"].items():
        print(f"  {metric:25s}: {value:+.2f}%")
    
    # Save results
    output_dir = Path(args.output_dir)
    evaluator.save_results(comparison, output_dir)
    evaluator.create_visualizations(comparison, output_dir / "visualizations")
    evaluator.save_example_predictions(
        baseline_preds, finetuned_preds, references, output_dir
    )
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. Review results in results/metrics/")
    print("  2. Run error analysis: python src/error_analysis.py")
    print("  3. Try inference: python src/inference.py")


if __name__ == "__main__":
    main()
