from .base_evaluator import BaseEvaluator
from datasets import load_dataset
from typing import Dict, Any
import re
from fuzzywuzzy import fuzz


def construct_prompt(data: dict):
    # cross-file prompt
    cross_file_prompt = f"# Repo Name: {data['repo_name']}\n"

    for snippet in data['context']:
        cross_file_prompt += f"# Path: {snippet['path']}\n{snippet['snippet']}" + "\n\n"
    
    # in-file prompt
    in_file_prompt = f"# Path: {data['file_path']}\n{data['import_statement']}\n{data['cropped_code']}\n"

    # normalize some empty lines
    cross_file_prompt = re.sub(r'\n{4,}', '\n\n', cross_file_prompt)
    in_file_prompt = re.sub(r'\n{4,}', '\n\n', in_file_prompt)

    return cross_file_prompt, in_file_prompt


def edit_similarity_score(predictions, ground_truths):
    """
    This function computes the average edit similarity score between the predicted codes and the ground truth codes. 
    It returns a float value between 0 and 1 indicating the degree of similarity between the predicted codes 
    and the ground truth codes, where a value of 1 means all the predicted codes are identical to their corresponding 
    ground truth codes and a value of 0 means none of the predicted codes are similar to their corresponding 
    ground truth codes.
    
    Args:
    predictions: list, predicted codes
    ground_truths: list, ground truth codes
    
    Returns:
    Float, the average edit similarity score between the predicted codes and the ground truth codes.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("The length of the predicted codes and the ground truth codes should be equal.")
    
    edit_sim = 0.0
    for pred, gt in zip(predictions, ground_truths):
        edit_sim += fuzz.ratio(pred, gt)
    
    return edit_sim / len(predictions) / 100


def get_first_line_not_comment(code:str):
    """
    This function gets the first line of code that is not a comment.

    Args:
    code: Str, the code

    Returns:
    Str, the first line of code that is not a comment or the first line of code if there is no line that is not a comment
    """

    # first remove the \n at the beginning of the code
    code = code.lstrip('\n')

    lines = code.split('\n')
    in_multiline_comment = False

    for line in lines:
        # if the line is empty, then skip
        if not line.strip():
            continue
        # if the line is a start of a multiline comment, then set the in_multiline_comment to True and skip
        if not in_multiline_comment and (line.strip().startswith('"""') or line.strip().startswith("'''")):
            in_multiline_comment = True
            continue
        # if the line is the end of a multiline comment, then set the in_multiline_comment to False and skip
        if in_multiline_comment and (line.strip().endswith('"""') or line.strip().endswith("'''")):
            in_multiline_comment = False
            continue
        # if the line is in a multiline comment, then skip
        if in_multiline_comment:
            continue
        # if the line is a single line comment, then skip
        if line.strip().startswith('#'):
            continue
        # if the line contains ```python or ``` , then skip
        if line.strip().startswith('```python') or line.strip().startswith('```'):
            continue
        # if the line is not a comment, then return the line
        return line
        
    # if we cannot find a line that is not a comment, then return the first line
    return lines[0]


class RepoBenchEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()
        self.max_tokens = 100
        self.truncate_input = True
        self.multiple_answers = False
        self.n_samples = 1000
        self.data = self.load_data()
        self.repobench = True
        self.name = "repobench"
        
    def load_data(self):
        dataset = load_dataset("tianyang/repobench_python_v1.1", split="cross_file_first")
        dataset = dataset.map(lambda x: {"prompt_A": construct_prompt(x)[0], "prompt_B": construct_prompt(x)[1] + "\nHere's the next line of code based on the context:", "answer": x["next_line"]}, remove_columns=dataset.column_names)
        dataset = dataset.filter(lambda x: len(x["prompt_A"]) < 5000)
        dataset = self.random_sample(dataset)
        return dataset

    def evaluate_item(self, item: Dict[str, Any], response: str):
        response = get_first_line_not_comment(response)
        if self.multiple_answers:
            answers = item['answers']
        else:
            answers = [item['answer']]
        exact_match_score = 0
        for answer in answers:
            try:
                scores = edit_similarity_score([response], [answer])
            except:
                continue
            exact_match_score = max(exact_match_score, scores)
        self.f1_total += exact_match_score
        self.f1_count += 1




    
