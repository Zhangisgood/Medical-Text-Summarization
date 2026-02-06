"""
Inference Script for Medical Text Summarization

This script provides a command-line interface for generating summaries
using the fine-tuned model.

Usage:
    python inference.py --model_path models/finetuned/final_model --input "Patient: ..."
"""

import os
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class MedicalSummarizer:
    """Inference pipeline for medical text summarization."""
    
    def __init__(self, model_path: str):
        """Initialize summarizer with trained model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model from: {model_path}")
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully!")
    
    def summarize(self, dialogue: str, max_length: int = 150, 
                  num_beams: int = 4, length_penalty: float = 0.6) -> str:
        """
        Generate summary for medical dialogue.
        
        Args:
            dialogue: Medical conversation text
            max_length: Maximum summary length
            num_beams: Number of beams for beam search
            length_penalty: Length penalty for generation
            
        Returns:
            Generated summary
        """
        # Format input with instruction
        instruction = "Summarize the following medical conversation between a patient and doctor. Focus on key symptoms, diagnosis, and treatment recommendations."
        input_text = f"{instruction}\n\nInput: {dialogue}"
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate summary
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        
        # Decode and return
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
    
    def interactive_mode(self):
        """Run interactive mode for continuous summarization."""
        print("\n" + "="*60)
        print("Medical Text Summarization - Interactive Mode")
        print("="*60)
        print("\nEnter medical dialogues to get summaries.")
        print("Type 'quit' or 'exit' to stop.\n")
        
        while True:
            print("-"*60)
            dialogue = input("\nEnter medical dialogue:\n> ")
            
            if dialogue.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not dialogue.strip():
                print("Please enter a valid dialogue.")
                continue
            
            print("\nGenerating summary...")
            summary = self.summarize(dialogue)
            
            print("\n" + "="*60)
            print("SUMMARY:")
            print("="*60)
            print(summary)
            print("="*60)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Generate medical summaries")
    parser.add_argument("--model_path", type=str, default="models/finetuned/final_model",
                       help="Path to fine-tuned model")
    parser.add_argument("--input", type=str, default=None,
                       help="Input dialogue to summarize")
    parser.add_argument("--max_length", type=int, default=150,
                       help="Maximum summary length")
    parser.add_argument("--num_beams", type=int, default=4,
                       help="Number of beams for beam search")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Initialize summarizer
    summarizer = MedicalSummarizer(args.model_path)
    
    if args.interactive:
        # Interactive mode
        summarizer.interactive_mode()
    elif args.input:
        # Single inference
        print("\n" + "="*60)
        print("INPUT:")
        print("="*60)
        print(args.input)
        
        print("\nGenerating summary...")
        summary = summarizer.summarize(
            args.input,
            max_length=args.max_length,
            num_beams=args.num_beams
        )
        
        print("\n" + "="*60)
        print("SUMMARY:")
        print("="*60)
        print(summary)
        print("="*60)
    else:
        # Default example
        example_dialogue = """Patient: I've been experiencing severe headaches for the past week, especially in the morning. They're accompanied by nausea and sensitivity to light. Sometimes I also feel dizzy.

Doctor: Based on your symptoms of severe morning headaches with nausea, photophobia, and dizziness, this appears to be migraine headaches. I recommend starting with over-the-counter pain relievers like ibuprofen at the onset of symptoms. Keep a headache diary to track triggers such as stress, certain foods, lack of sleep, or hormonal changes. If symptoms persist or worsen over the next two weeks, we may need to prescribe preventive medication. Also, ensure you're staying hydrated and maintaining regular sleep patterns."""
        
        print("\n" + "="*60)
        print("EXAMPLE DIALOGUE:")
        print("="*60)
        print(example_dialogue)
        
        print("\nGenerating summary...")
        summary = summarizer.summarize(example_dialogue)
        
        print("\n" + "="*60)
        print("GENERATED SUMMARY:")
        print("="*60)
        print(summary)
        print("="*60)
        
        print("\nTip: Use --input to provide your own dialogue")
        print("     Use --interactive for continuous summarization")


if __name__ == "__main__":
    main()
