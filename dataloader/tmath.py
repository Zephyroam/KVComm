from .base_evaluator import BaseEvaluator
from datasets import load_dataset
from rouge import Rouge
import os
import glob
from typing import Dict, Any


class TMathEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()
        self.max_tokens = 256
        self.truncate_input = True
        self.multiple_answers = False
        self.n_samples = 300
        self.data = self.load_data()
        self.rouge = Rouge()
        self.tmath = True
        self.name = "tmath"
        
    def load_data(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(script_dir, "data", "TMATH")
        all_files = []
        for split in ["hint_algebra", "hint_geometry", "hint_number_theory", 
                    "hint_intermediate_algebra", "hint_prealgebra", 
                    "hint_precalculus", "hint_counting_and_probability"]:
            files = glob.glob(f"{dataset_path}/{split}/*.json")
            all_files.extend(files)
        dataset = load_dataset("json", data_files=all_files, split="train")
        dataset = self.random_sample(dataset)
        dataset = dataset.rename_column("socratic_questions", "prompt_A")
        dataset = dataset.rename_column("problem", "prompt_B")
        dataset = dataset.rename_column("solution", "answer")
        return dataset

    def evaluate_item(self, item: Dict[str, Any], response: str):
        if self.multiple_answers:
            answers = item['answers']
        else:
            answers = [item['answer']]
        rouge_score = 0
        for answer in answers:
            try:
                scores = self.rouge.get_scores(response, answer)[0]
            except:
                continue
            rouge_score = max(rouge_score, scores["rouge-l"]["r"])
        self.f1_total += rouge_score
        self.f1_count += 1




    
