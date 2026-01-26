from .base_evaluator import BaseEvaluator
from datasets import load_dataset
import os


class CountriesEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()
        self.max_tokens = 5
        self.truncate_input = False
        self.multiple_answers = False
        self.data = self.load_data()
        self.name = "countries"
        
    def load_data(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(script_dir, "data", "countries.jsonl")
        dataset = load_dataset("json", data_files=dataset_path)["train"]
        dataset = dataset.remove_columns(["id", "person"])
        dataset = dataset.rename_column("country", "answer")
        dataset = dataset.map(lambda x: {"prompt_B": x['prompt_B'] + "You are required to extrapolate the country from the context."})
        return dataset