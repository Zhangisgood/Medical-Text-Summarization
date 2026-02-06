"""
Error Analysis Script for Medical Text Summarization

This script performs detailed error analysis on model predictions,
categorizing errors and identifying patterns for improvement.

Usage:
    python error_analysis.py --model_path models/finetuned/final_model
"""

import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
import evaluate
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from rouge_score import rouge_scorer


class ErrorAnalyzer:
    """Analyzes model errors and categorizes failure patterns."""
    
    def __init__(self, model_path: str):
        """Initialize analyzer with model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        print(f"Loading model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    def generate_summary(self, input_text: str) -> str:
        """Generate summary for input text."""
        inputs = self.tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=150,
                num_beams=4,
                length_penalty=0.6,
                early_stopping=True,
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def compute_rouge_score(self, prediction: str, reference: str) -> float:
        """Compute average ROUGE score."""
        scores = self.rouge_scorer.score(reference, prediction)
        avg_score = (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
        return avg_score
    
    def categorize_error(self, prediction: str, reference: str, input_text: str) -> str:
        """Categorize the type of error."""
        pred_len = len(prediction.split())
        ref_len = len(reference.split())
        
        # Length-based errors
        if pred_len < ref_len * 0.5:
            return "Too Brief"
        elif pred_len > ref_len * 1.5:
            return "Too Verbose"
        
        # Content-based errors
        ref_words = set(reference.lower().split())
        pred_words = set(prediction.lower().split())
        
        overlap = len(ref_words & pred_words) / len(ref_words) if ref_words else 0
        
        if overlap < 0.3:
            return "Factual Error"
        elif overlap < 0.5:
            return "Missing Information"
        else:
            return "Formatting Error"
    
    def analyze_errors(self, test_dataset, threshold: float = 0.5) -> Tuple[List[Dict], Dict]:
        """Analyze errors in test dataset."""
        print(f"\nAnalyzing errors (threshold: {threshold})...")
        
        errors = []
        error_categories = Counter()
        all_scores = []
        
        for idx, example in enumerate(tqdm(test_dataset, desc="Analyzing predictions")):
            # Generate prediction
            input_text = f"{example['instruction']}\n\nInput: {example['input']}"
            prediction = self.generate_summary(input_text)
            reference = example['output']
            
            # Compute score
            score = self.compute_rouge_score(prediction, reference)
            all_scores.append(score)
            
            # Identify errors
            if score < threshold:
                category = self.categorize_error(prediction, reference, example['input'])
                error_categories[category] += 1
                
                errors.append({
                    'index': idx,
                    'input': example['input'][:200] + "..." if len(example['input']) > 200 else example['input'],
                    'reference': reference,
                    'prediction': prediction,
                    'score': round(score, 4),
                    'category': category,
                })
        
        # Calculate statistics
        stats = {
            'total_examples': len(test_dataset),
            'errors_found': len(errors),
            'error_rate': round(len(errors) / len(test_dataset) * 100, 2),
            'average_score': round(np.mean(all_scores), 4),
            'median_score': round(np.median(all_scores), 4),
            'min_score': round(np.min(all_scores), 4),
            'max_score': round(np.max(all_scores), 4),
        }
        
        return errors, dict(error_categories), stats
    
    def create_visualizations(self, error_categories: Dict, stats: Dict, output_dir: Path):
        """Create error analysis visualizations."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        sns.set_style("whitegrid")
        
        # 1. Error category distribution
        if error_categories:
            fig, ax = plt.subplots(figsize=(10, 6))
            categories = list(error_categories.keys())
            counts = list(error_categories.values())
            
            bars = ax.bar(categories, counts, alpha=0.7, color='coral')
            ax.set_xlabel('Error Category', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title('Error Category Distribution', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=10)
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_dir / "error_categories.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Error rate pie chart
        fig, ax = plt.subplots(figsize=(8, 8))
        sizes = [stats['total_examples'] - stats['errors_found'], stats['errors_found']]
        labels = ['Correct', 'Errors']
        colors = ['lightgreen', 'lightcoral']
        explode = (0, 0.1)
        
        ax.pie(sizes, explode=explode, labels=labels, colors=colors,
               autopct='%1.1f%%', shadow=True, startangle=90)
        ax.set_title(f"Error Rate: {stats['error_rate']}%", fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / "error_rate.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nVisualizations saved to: {output_dir}")
    
    def generate_report(self, errors: List[Dict], error_categories: Dict, 
                       stats: Dict, output_dir: Path):
        """Generate comprehensive error analysis report."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = output_dir / "error_analysis_report.txt"
        with open(report_path, "w") as f:
            f.write("="*80 + "\n")
            f.write("Medical Text Summarization - Error Analysis Report\n")
            f.write("="*80 + "\n\n")
            
            # Statistics
            f.write("OVERALL STATISTICS\n")
            f.write("-"*80 + "\n")
            for key, value in stats.items():
                f.write(f"  {key:25s}: {value}\n")
            
            # Error categories
            f.write("\nERROR CATEGORIES\n")
            f.write("-"*80 + "\n")
            for category, count in sorted(error_categories.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / stats['errors_found']) * 100 if stats['errors_found'] > 0 else 0
                f.write(f"  {category:25s}: {count:4d} ({percentage:5.1f}%)\n")
            
            # Error examples
            f.write("\nERROR EXAMPLES (Top 10 Worst Cases)\n")
            f.write("-"*80 + "\n\n")
            
            sorted_errors = sorted(errors, key=lambda x: x['score'])[:10]
            for i, error in enumerate(sorted_errors, 1):
                f.write(f"Example {i} (Score: {error['score']:.4f}, Category: {error['category']})\n")
                f.write("-"*80 + "\n")
                f.write(f"Input:\n{error['input']}\n\n")
                f.write(f"Reference:\n{error['reference']}\n\n")
                f.write(f"Prediction:\n{error['prediction']}\n\n")
                f.write("="*80 + "\n\n")
            
            # Improvement suggestions
            f.write("SUGGESTED IMPROVEMENTS\n")
            f.write("-"*80 + "\n")
            
            suggestions = self.generate_suggestions(error_categories, stats)
            for i, suggestion in enumerate(suggestions, 1):
                f.write(f"{i}. {suggestion}\n")
        
        # Save errors as JSON
        errors_json_path = output_dir / "errors.json"
        with open(errors_json_path, "w") as f:
            json.dump({
                'statistics': stats,
                'error_categories': error_categories,
                'errors': errors[:50],  # Save top 50 errors
            }, f, indent=2)
        
        print(f"\nReport saved to:")
        print(f"  - {report_path}")
        print(f"  - {errors_json_path}")
    
    def generate_suggestions(self, error_categories: Dict, stats: Dict) -> List[str]:
        """Generate improvement suggestions based on error patterns."""
        suggestions = []
        
        if not error_categories:
            suggestions.append("Model performance is excellent! Consider testing on more challenging cases.")
            return suggestions
        
        # Get most common error
        most_common = max(error_categories.items(), key=lambda x: x[1])
        
        if most_common[0] == "Too Brief":
            suggestions.append(
                "Increase minimum length constraint during generation or adjust length_penalty parameter."
            )
            suggestions.append(
                "Add more examples of comprehensive summaries to training data."
            )
        
        elif most_common[0] == "Too Verbose":
            suggestions.append(
                "Decrease maximum length or increase length_penalty to encourage conciseness."
            )
            suggestions.append(
                "Fine-tune with more concise reference summaries."
            )
        
        elif most_common[0] == "Factual Error":
            suggestions.append(
                "Increase training epochs or learning rate to improve content understanding."
            )
            suggestions.append(
                "Augment training data with more diverse medical scenarios."
            )
            suggestions.append(
                "Consider using a larger base model (e.g., flan-t5-large)."
            )
        
        elif most_common[0] == "Missing Information":
            suggestions.append(
                "Adjust beam search parameters (increase num_beams) for better coverage."
            )
            suggestions.append(
                "Add examples emphasizing key information extraction to training set."
            )
        
        else:  # Formatting Error
            suggestions.append(
                "Refine prompt templates to emphasize desired output format."
            )
            suggestions.append(
                "Add post-processing rules to standardize output format."
            )
        
        # General suggestions
        if stats['error_rate'] > 30:
            suggestions.append(
                "Consider collecting more training data or using data augmentation techniques."
            )
        
        suggestions.append(
            "Implement human-in-the-loop evaluation for critical cases."
        )
        
        return suggestions


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Analyze model errors")
    parser.add_argument("--model_path", type=str, default="models/finetuned/final_model",
                       help="Path to fine-tuned model")
    parser.add_argument("--data_dir", type=str, default="data/processed/dataset_dict",
                       help="Directory containing processed datasets")
    parser.add_argument("--output_dir", type=str, default="results/error_analysis",
                       help="Output directory for analysis results")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="ROUGE score threshold for error identification")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Medical Text Summarization - Error Analysis")
    print("="*60)
    
    # Load test dataset
    print(f"\nLoading test dataset from: {args.data_dir}")
    dataset_dict = load_from_disk(args.data_dir)
    test_dataset = dataset_dict["test"]
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize analyzer
    analyzer = ErrorAnalyzer(args.model_path)
    
    # Analyze errors
    errors, error_categories, stats = analyzer.analyze_errors(test_dataset, args.threshold)
    
    # Print summary
    print("\n" + "="*60)
    print("ERROR ANALYSIS SUMMARY")
    print("="*60)
    
    print("\nStatistics:")
    for key, value in stats.items():
        print(f"  {key:25s}: {value}")
    
    print("\nError Categories:")
    for category, count in sorted(error_categories.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / stats['errors_found']) * 100 if stats['errors_found'] > 0 else 0
        print(f"  {category:25s}: {count:4d} ({percentage:5.1f}%)")
    
    # Generate outputs
    output_dir = Path(args.output_dir)
    analyzer.generate_report(errors, error_categories, stats, output_dir)
    analyzer.create_visualizations(error_categories, stats, output_dir / "visualizations")
    
    # Print suggestions
    print("\nSuggested Improvements:")
    suggestions = analyzer.generate_suggestions(error_categories, stats)
    for i, suggestion in enumerate(suggestions, 1):
        print(f"  {i}. {suggestion}")
    
    print("\n" + "="*60)
    print("Error Analysis Complete!")
    print("="*60)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
