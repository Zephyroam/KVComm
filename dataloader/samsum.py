from .base_evaluator import BaseEvaluator
from datasets import load_dataset
from rouge import Rouge

def split_dialogue(item):
    # split dialogue into two halves, by lines
    dialogue_lines = item["dialogue"].strip().split("\n")
    mid_point = len(dialogue_lines) // 2
    prompt_A = "\n".join(dialogue_lines[:mid_point]).strip()
    prompt_B = "\n".join(dialogue_lines[mid_point:]).strip()
    return {"prompt_A": prompt_A, "prompt_B": prompt_B, "answer": item["summary"]}

class SAMSumEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()
        self.max_tokens = 300
        self.truncate_input = True
        self.multiple_answers = False
        self.data = self.load_data()
        self.name = "samsum"
        self.sasum = True
        self.rouge = Rouge()
        
    def load_data(self):
        dataset = load_dataset("knkarthick/samsum")["test"]
        dataset = dataset.map(lambda x: split_dialogue(x))
        return dataset

    def evaluate_item(self, item, response: str):
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
