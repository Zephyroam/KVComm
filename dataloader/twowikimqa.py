from .base_evaluator import BaseEvaluator
from datasets import load_dataset
import re

def split_passages(text, num_parts=2):
    pattern = re.compile(r'(Passage\s+\d+:)')
    matches = list(pattern.finditer(text))

    if not matches:
        raise ValueError("No passages found in the input text.")

    first_passage_start = matches[0].start()
    head = text[:first_passage_start]

    passages = []
    for i, m in enumerate(matches):
        start = m.start()
        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)
        passages.append(text[start:end])

    last_passage_full_end = passages[-1].rfind('\n')
    if last_passage_full_end == -1:
        last_passage_full_end = len(passages[-1])
    last_passage_text = passages[-1]
    last_occ_idx = text.rfind(last_passage_text)
    tail_start = last_occ_idx + len(last_passage_text)
    tail = text[tail_start:]

    if tail and not tail.startswith('\n'):
        tail = '\n' + tail

    total = len(passages)
    parts = []
    for i in range(num_parts):
        start_idx = (total * i) // num_parts
        end_idx = (total * (i + 1)) // num_parts
        body = ''.join(passages[start_idx:end_idx])
        part_text = head + body + tail
        parts.append(part_text)

    return parts


class TwoWikiMQAEvaluator(BaseEvaluator):
    def __init__(self, multi_agent=False):
        super().__init__()
        self.max_tokens = 32
        self.truncate_input = True
        self.multiple_answers = True
        if multi_agent:
            self.data = self.load_data_multi_sender()
        else:
            self.data = self.load_data_single_sender()
        self.name = "2wikimqa"
        self.multi_agent = multi_agent

    def load_data_single_sender(self):
        dataset = load_dataset('Xnhyacinth/LongBench', split='test', name='2wikimqa')
        dataset = dataset.map(lambda x: {
            "prompt_A": x["context"], 
            "prompt_B": x["question"], 
        })
        return dataset

    def load_data_multi_sender(self):
        dataset = load_dataset('Xnhyacinth/LongBench', split='test', name='2wikimqa')
        dataset = dataset.map(lambda x: {'passage_parts': split_passages(x['context'])})
        dataset = dataset.map(lambda x: {
            "prompt_A1": x["passage_parts"][0],
            "prompt_A2": x["passage_parts"][1],
            "prompt_B": x["question"], 
        })
        return dataset