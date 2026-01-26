from utils import f1_match
from typing import Dict, Any
import random

class BaseEvaluator:
    def __init__(self, random_state=42):
        self.index = 0
        self.f1_total = 0.0
        self.f1_count = 0
        self.max_tokens = None
        self.truncate_input = None
        self.multiple_answers = None
        self.n_samples = None
        self.random_state = random_state

    def load_data(self):
        pass

    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
        self.index = 0
        self.f1_total = 0.0
        self.f1_count = 0
        return self
    
    def __next__(self):
        if self.index < len(self):
            value = self.data[self.index]
            self.index += 1
            return value
        else:
            raise StopIteration

    def evaluate_item(self, item: Dict[str, Any], response: str):
        if self.multiple_answers:
            answers = item['answers']
        else:
            answers = [item['answer']]
        f1_score = 0
        for answer in answers:
            f1_score = max(f1_score, f1_match(answer, response))
        self.f1_total += f1_score
        self.f1_count += 1
    
    def get_result(self) -> float:
        return self.f1_total / self.f1_count if self.f1_count > 0 else 0.0
    
    def random_sample(self, dataset):
        if self.n_samples is None:
            return dataset
        else:
            return dataset.shuffle(seed=self.random_state).select(range(self.n_samples))