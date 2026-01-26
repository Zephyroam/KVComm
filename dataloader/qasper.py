from .base_evaluator import BaseEvaluator
from datasets import load_dataset


class QaSperEvaluator(BaseEvaluator):
    def __init__(self, n_samples=500):
        super().__init__()
        self.max_tokens = 128
        self.truncate_input = True
        self.multiple_answers = False
        self.n_samples = n_samples
        self.data = self.load_data()
        self.name = "qasper"
        
    def load_data(self):
        dataset = load_dataset("tau/scrolls", name="qasper", trust_remote_code=True)["validation"]
        dataset = self.random_sample(dataset)
        dataset = dataset.map(lambda x: {
            "prompt_A": x["input"][x["input"].index("\n\n")+2:].strip(), 
            "prompt_B": x["input"][:x["input"].index("\n\n")].strip(), 
            "answer": x["output"],
        })
        return dataset
