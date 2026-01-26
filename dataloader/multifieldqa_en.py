from .base_evaluator import BaseEvaluator
from datasets import load_dataset


class MultiFieldQAEnEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()
        self.max_tokens = 64
        self.truncate_input = True
        self.multiple_answers = True
        self.data = self.load_data()
        self.name = "multifieldqa_en"
        
    def load_data(self):
        dataset = load_dataset('Xnhyacinth/LongBench', split='test', name='multifieldqa_en')
        dataset = dataset.map(lambda x: {
            "prompt_A": x["context"], 
            "prompt_B": x["question"], 
        })
        return dataset
