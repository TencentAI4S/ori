import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
import re
import random
import argparse
import os
import logging
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PeptideGenerator:
    def __init__(self, model_path: str, tokenizer_path: str):
        logging.info("Initializing PeptideGenerator...")
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path, pad_token="[PAD]", mask_token="[MASK]")
        self.model = GPT2LMHeadModel.from_pretrained(model_path).to('cuda' if torch.cuda.is_available() else 'cpu').eval()
        print("model parameters:", sum(p.numel() for p in self.model.parameters()))

    def generate_text(self, prompt: str, max_length: int = 512, num_sequences: int = 3) -> List[str]:
        logging.info(f"Generating text with prompt: {prompt}")
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to('cuda' if torch.cuda.is_available() else 'cpu')
        
        logging.info("Generating sequences...")
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=num_sequences,
                no_repeat_ngram_size=4,
                temperature=0.95,
                top_k=10,
                top_p=0.95,
                num_beams=num_sequences,
                num_beam_groups=num_sequences,
                diversity_penalty=0.5,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        generated_sequences = [self.tokenizer.decode(output[i], skip_special_tokens=False) for i in range(num_sequences)]
        return generated_sequences

    @staticmethod
    def extract_peptide(generated_text: str) -> Optional[str]:
        logging.debug(f"Extracting peptide from generated data...")
        match = re.search(r'<peptide>(.*?)<end>', generated_text)
        return PeptideGenerator.clean_sequence(match.group(1).strip()) if match else None

    @staticmethod
    def clean_sequence(sequence: str) -> str:
        cleaned_sequence = re.sub(r'\[SEP\]|\[CLS\]|<aligned>|<protein>|<peptide>|<end>', '', sequence).strip()
        logging.debug(f"Cleaned sequence...")
        return cleaned_sequence

    def generate_valid_peptides(self, prompt: str, max_length: int = 512, num_sequences: int = 3) -> List[str]:
        logging.info("Generating valid peptide sequences...")
        valid_peptides = []
        peptide_sequences = self.generate_text(prompt, max_length, num_sequences)
        
        for peptide_sequence in peptide_sequences:
            cleaned_sequence = self.extract_peptide(peptide_sequence)
            if cleaned_sequence and 2 <= len(cleaned_sequence) <= 102 and 'X' not in cleaned_sequence and cleaned_sequence not in valid_peptides:
                valid_peptides.append(cleaned_sequence)

        logging.info(f"Total valid peptides generated: {len(valid_peptides)}")
        return valid_peptides

def main(model_path: str, tokenizer_path: str, prompt: str, max_length: int, num_sequences: int):
    logging.info("Starting peptide generation process...")
    generator = PeptideGenerator(model_path, tokenizer_path)
    valid_peptides = generator.generate_valid_peptides(prompt, max_length, num_sequences)
    logging.info(f"Generated Peptide Sequences: {valid_peptides}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate valid peptide sequences using a pre-trained model.")
    parser.add_argument("--model_path", type=str, default='../model_weights/PepGPT', help="Path to the pre-trained model.")
    parser.add_argument("--tokenizer_path", type=str, default='../model_weights/PepGPT/tokenizer.json', help="Path to the tokenizer file.")
    parser.add_argument("--prompt", type=str, default='MKEQDSEEELIEAFKVFDRDGNGLISAAELRHVMTNLGEKLTDDEVDEMIREADIDGDGHINYEEFVRMMVSK', help="Prompt for generating peptides.")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum length of generated sequences.")
    parser.add_argument("--num_sequences", type=int, default=100, help="Number of sequences to generate.")

    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path '{args.model_path}' does not exist.")
    if not os.path.exists(args.tokenizer_path):
        raise FileNotFoundError(f"Tokenizer path '{args.tokenizer_path}' does not exist.")

    input_prompt = f"<aligned><protein>{args.prompt}<peptide>"

    main(args.model_path, args.tokenizer_path, input_prompt, args.max_length, args.num_sequences)