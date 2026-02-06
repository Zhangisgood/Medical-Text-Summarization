"""
Data Preparation Script for Medical Text Summarization

This script downloads, preprocesses, and formats medical dialogue data
for fine-tuning a summarization model.

Usage:
    python data_preparation.py [--dataset_name DATASET] [--max_samples N]
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from tqdm import tqdm
import re


class MedicalDataPreprocessor:
    """Handles medical dialogue data preprocessing and formatting."""
    
    def __init__(self, tokenizer_name: str = "google/flan-t5-base"):
        """Initialize preprocessor with tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_input_length = 512
        self.max_target_length = 150
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep medical terminology
        text = re.sub(r'[^\w\s\-.,;:?!()/]', '', text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text
    
    def format_dialogue(self, dialogue: str, summary: str) -> Dict[str, str]:
        """Format dialogue and summary into instruction format."""
        instruction = "Summarize the following medical conversation between a patient and doctor. Focus on key symptoms, diagnosis, and treatment recommendations."
        
        formatted = {
            "instruction": instruction,
            "input": self.clean_text(dialogue),
            "output": self.clean_text(summary),
        }
        return formatted
    
    def load_medical_dataset(self, dataset_name: str = "medical", max_samples: int = None) -> DatasetDict:
        """
        Load medical dialogue dataset.
        
        For this project, we'll use a combination of publicly available datasets.
        In a real scenario, you would use actual medical dialogue data.
        """
        print(f"Loading dataset: {dataset_name}")
        
        try:
            # Try to load from Hugging Face Hub
            # Using a medical dialogue dataset or creating synthetic data
            dataset = load_dataset("ruslanmv/ai-medical-chatbot", split="train")
            
            # Process the dataset
            processed_data = []
            for idx, example in enumerate(tqdm(dataset, desc="Processing examples")):
                if max_samples and idx >= max_samples:
                    break
                
                # Extract dialogue and create summary
                # This dataset has 'Patient' and 'Doctor' fields
                if 'Patient' in example and 'Doctor' in example:
                    dialogue = f"Patient: {example['Patient']}\nDoctor: {example['Doctor']}"
                    # For this dataset, we'll create summaries from doctor's response
                    # In real scenario, you'd have ground truth summaries
                    summary = self._create_summary(example['Doctor'])
                    
                    formatted = self.format_dialogue(dialogue, summary)
                    processed_data.append(formatted)
            
            print(f"Processed {len(processed_data)} examples")
            
        except Exception as e:
            print(f"Could not load dataset from Hub: {e}")
            print("Creating synthetic medical dialogue dataset...")
            processed_data = self._create_synthetic_dataset(max_samples or 1000)
        
        # Convert to Dataset
        df = pd.DataFrame(processed_data)
        dataset = Dataset.from_pandas(df)
        
        return dataset
    
    def _create_summary(self, doctor_response: str, max_length: int = 100) -> str:
        """Create a concise summary from doctor's response."""
        # Simple summarization: take first few sentences
        sentences = doctor_response.split('.')
        summary = '. '.join(sentences[:2]) + '.'
        return summary[:max_length]
    
    def _create_synthetic_dataset(self, num_samples: int) -> List[Dict[str, str]]:
        """Create synthetic medical dialogue dataset for demonstration."""
        
        # Template-based synthetic data
        templates = [
            {
                "dialogue": "Patient: I've been experiencing severe headaches for the past week, especially in the morning. They're accompanied by nausea and sensitivity to light.\nDoctor: Based on your symptoms, this could be migraine headaches. I recommend starting with over-the-counter pain relievers and keeping a headache diary. If symptoms persist, we may need to prescribe preventive medication. Also, try to identify and avoid potential triggers like stress, certain foods, or lack of sleep.",
                "summary": "Patient reports severe morning headaches with nausea and photophobia for one week. Diagnosis: Possible migraine. Treatment: OTC pain relievers, headache diary, trigger avoidance. Follow-up if symptoms persist."
            },
            {
                "dialogue": "Patient: I have a persistent cough that's lasted for three weeks. It's worse at night and I'm producing yellow mucus.\nDoctor: A cough lasting three weeks with yellow sputum suggests a possible bacterial respiratory infection. I'll prescribe a course of antibiotics and a cough suppressant. Make sure to stay hydrated and get plenty of rest. If the cough doesn't improve in 5-7 days or if you develop fever or difficulty breathing, please return immediately.",
                "summary": "Patient presents with 3-week productive cough, worse at night, yellow sputum. Diagnosis: Suspected bacterial respiratory infection. Treatment: Antibiotics, cough suppressant, hydration, rest. Return if no improvement in 5-7 days or symptoms worsen."
            },
            {
                "dialogue": "Patient: I've noticed my blood pressure has been consistently high when I check it at home, around 150/95. I feel fine otherwise.\nDoctor: Elevated blood pressure readings of 150/95 are concerning and indicate hypertension. Even without symptoms, high blood pressure can damage your heart and blood vessels over time. I recommend lifestyle modifications including reducing salt intake, regular exercise, and weight management. I'm also prescribing a low-dose blood pressure medication. We'll monitor your response and adjust as needed.",
                "summary": "Patient reports home BP readings of 150/95, asymptomatic. Diagnosis: Hypertension. Treatment plan: Lifestyle modifications (low sodium diet, exercise, weight management) plus antihypertensive medication. Follow-up for monitoring and adjustment."
            },
            {
                "dialogue": "Patient: I have a rash on my arms that's very itchy. It appeared after I went hiking last weekend.\nDoctor: The timing and location suggest this could be contact dermatitis, possibly from poison ivy or another plant. The rash should resolve on its own in 1-2 weeks. I recommend applying hydrocortisone cream twice daily and taking an oral antihistamine for the itching. Avoid scratching to prevent infection. If it spreads significantly or shows signs of infection, please come back.",
                "summary": "Patient presents with itchy arm rash post-hiking. Diagnosis: Contact dermatitis, likely plant-related. Treatment: Topical hydrocortisone, oral antihistamine, avoid scratching. Expected resolution in 1-2 weeks. Return if spreading or signs of infection."
            },
            {
                "dialogue": "Patient: I've been feeling extremely tired lately, even after a full night's sleep. I also feel cold all the time and my hair seems to be thinning.\nDoctor: Your symptoms of fatigue, cold intolerance, and hair thinning are classic signs of hypothyroidism. I'd like to order blood tests to check your thyroid hormone levels. If confirmed, this is easily treatable with thyroid hormone replacement medication. We'll start with the blood work and discuss treatment options once we have the results.",
                "summary": "Patient reports persistent fatigue, cold intolerance, and hair thinning despite adequate sleep. Suspected diagnosis: Hypothyroidism. Plan: Thyroid function tests (TSH, T4). Treatment pending lab results, likely thyroid hormone replacement if confirmed."
            },
        ]
        
        # Replicate templates to create desired number of samples
        synthetic_data = []
        for i in range(num_samples):
            template = templates[i % len(templates)]
            # Add slight variations
            formatted = self.format_dialogue(template["dialogue"], template["summary"])
            synthetic_data.append(formatted)
        
        print(f"Created {len(synthetic_data)} synthetic examples")
        return synthetic_data
    
    def tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """Tokenize dataset for model training."""
        
        def tokenize_function(examples):
            # Combine instruction and input
            inputs = [
                f"{inst}\n\nInput: {inp}" 
                for inst, inp in zip(examples["instruction"], examples["input"])
            ]
            
            model_inputs = self.tokenizer(
                inputs,
                max_length=self.max_input_length,
                truncation=True,
                padding="max_length",
            )
            
            # Tokenize targets
            labels = self.tokenizer(
                examples["output"],
                max_length=self.max_target_length,
                truncation=True,
                padding="max_length",
            )
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset",
        )
        
        return tokenized
    
    def split_dataset(self, dataset: Dataset, train_ratio: float = 0.8, 
                     val_ratio: float = 0.1) -> DatasetDict:
        """Split dataset into train, validation, and test sets."""
        
        # First split: train vs (val + test)
        train_test = dataset.train_test_split(test_size=1-train_ratio, seed=42)
        
        # Second split: val vs test
        val_test_ratio = val_ratio / (1 - train_ratio)
        val_test = train_test["test"].train_test_split(test_size=1-val_test_ratio, seed=42)
        
        dataset_dict = DatasetDict({
            "train": train_test["train"],
            "validation": val_test["train"],
            "test": val_test["test"],
        })
        
        return dataset_dict
    
    def save_dataset(self, dataset_dict: DatasetDict, output_dir: str):
        """Save processed dataset to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON for easy inspection
        for split_name, split_data in dataset_dict.items():
            split_path = output_path / f"{split_name}.json"
            split_data.to_json(split_path)
            print(f"Saved {split_name} split to {split_path} ({len(split_data)} examples)")
        
        # Also save the entire dataset dict
        dataset_dict.save_to_disk(str(output_path / "dataset_dict"))
        print(f"Saved complete dataset to {output_path / 'dataset_dict'}")
        
        # Save statistics
        stats = {
            "train_size": len(dataset_dict["train"]),
            "validation_size": len(dataset_dict["validation"]),
            "test_size": len(dataset_dict["test"]),
            "total_size": len(dataset_dict["train"]) + len(dataset_dict["validation"]) + len(dataset_dict["test"]),
        }
        
        with open(output_path / "statistics.json", "w") as f:
            json.dump(stats, f, indent=2)
        
        print("\nDataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Prepare medical dialogue dataset")
    parser.add_argument("--dataset_name", type=str, default="medical",
                       help="Name of dataset to load")
    parser.add_argument("--max_samples", type=int, default=1000,
                       help="Maximum number of samples to process")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                       help="Output directory for processed data")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Medical Text Summarization - Data Preparation")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = MedicalDataPreprocessor()
    
    # Load dataset
    print("\n[1/4] Loading dataset...")
    dataset = preprocessor.load_medical_dataset(
        dataset_name=args.dataset_name,
        max_samples=args.max_samples
    )
    
    # Split dataset
    print("\n[2/4] Splitting dataset...")
    dataset_dict = preprocessor.split_dataset(dataset)
    
    # Tokenize dataset
    print("\n[3/4] Tokenizing dataset...")
    tokenized_dict = DatasetDict({
        split: preprocessor.tokenize_dataset(data)
        for split, data in dataset_dict.items()
    })
    
    # Save dataset
    print("\n[4/4] Saving processed dataset...")
    preprocessor.save_dataset(tokenized_dict, args.output_dir)
    
    print("\n" + "="*60)
    print("Data preparation complete!")
    print("="*60)
    print(f"\nProcessed data saved to: {args.output_dir}")
    print("\nNext steps:")
    print("  1. Review the data in data/processed/")
    print("  2. Run training: python src/train.py")


if __name__ == "__main__":
    main()
