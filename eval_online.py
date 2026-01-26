import torch
import torch.nn.functional as F
import logging
from tqdm import tqdm
import wandb
from layer_importance import calc_layer_importance
from collections import defaultdict
from layer_importance import get_top_layers
import numpy as np

QA_INSTRUCTION = "Directly answer the question based on the context passage, no explanation is needed."
MATH_INSTRUCTION = "Answer the math problem step by step."
CODE_INSTRUCTION = "Complete ONLY THE NEXT LINE of the code snippet based on the context."
SUMMARIZE_INSTRUCTION = "Summarize the following content concisely with one sentence."

COMMUNICATION_QA_MSG_TEMPLATE_A = "Instruction: {instruction} Context: {context}"
COMMUNICATION_QA_MSG_TEMPLATE_B = "Instruction: {instruction} Question: {question}"
COMMUNICATION_MATH_MSG_TEMPLATE_A = "Instruction: {instruction} Hint: {hint}"
COMMUNICATION_MATH_MSG_TEMPLATE_B = "Instruction: {instruction} Question: {question}"
COMMUNICATION_CODE_MSG_TEMPLATE_A = "Instruction: {instruction} Context: {context}"
COMMUNICATION_CODE_MSG_TEMPLATE_B = "Instruction: {instruction} Code Snippet: {code_snippet}"
COMMUNICATION_SUMMARIZE_MSG_TEMPLATE_A = "Instruction: {instruction} Content part 1: {content_part_1}"
COMMUNICATION_SUMMARIZE_MSG_TEMPLATE_B = "Instruction: {instruction} Content part 2: {content_part_2}"

THINK_MODEL_LIST = ["deepseek-ai/DeepSeek-R1-Distill-Llama-8B"]

def is_think_model(model):
    for think_model in THINK_MODEL_LIST:
        if think_model.lower() == model.name.lower():
            return True
    return False

def apply_chat_template(evaluator, tokenizer, msg, model, context=False):
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": msg}],
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    if is_think_model(model):
        think_model_prefix = "</think>\n\n"
        if not context and evaluator.name not in ["tipsheets", "repobench"]:
            # for tipsheets, we do not add "The answer is: " as there is already
            # an answer prefix in the data
            if evaluator.name == "countries":
                think_model_prefix += "The only country is:"
            else:
                think_model_prefix += "The answer is: "
        
        if context:
            think_token_id = tokenizer.convert_tokens_to_ids("<think>")
            # remove the think token from the input ids
            input_ids = input_ids[input_ids != think_token_id].unsqueeze(0)
        else:
            end_think_token_id = tokenizer.encode(think_model_prefix, add_special_tokens=False)
            input_ids = torch.cat([input_ids, torch.tensor([end_think_token_id], device=model.device)], dim=-1)
    return input_ids



class CommunicationEvaluator:
    def __init__(self, evaluator, tokenizer, use_wandb, max_input_length, cfg):
        self.evaluator = evaluator
        self.tokenizer = tokenizer
        self.use_wandb = use_wandb
        self.max_input_length = max_input_length
        self.name = "skyline"
        self.generate_args = {
            "max_new_tokens": self.evaluator.max_tokens,
            "temperature": 1.0,
            "num_beams": 1,
            "top_p": None,
            "top_k": None,
            "do_sample": False,
            "pad_token_id": self.tokenizer.eos_token_id
        }
        self.name = "communication"
        self.layer_importance_total = defaultdict(list)
        self.cfg = cfg
    
    def get_response(self, output, context_length, truncate_response=True):
        if truncate_response:
            response = self.tokenizer.decode(output[context_length:], skip_special_tokens=True)
        else:
            response = self.tokenizer.decode(output, skip_special_tokens=True)
        return response

    def truncate_input(self, input_ids_A, input_ids_B):
        if input_ids_A.shape[-1] + input_ids_B.shape[-1] > self.max_input_length and self.evaluator.truncate_input:
            half = int((self.max_input_length - input_ids_B.shape[-1]) / 2)
            input_ids_A = torch.cat([input_ids_A[:, :half], input_ids_A[:, -half:]], dim=-1)
        return input_ids_A, input_ids_B

    def prepare_input_ids(self, item, model_A, model_B):
        if hasattr(self.evaluator, "tmath"):
            msg_A = COMMUNICATION_MATH_MSG_TEMPLATE_A.format(instruction=MATH_INSTRUCTION, hint=item["prompt_A"])
        elif hasattr(self.evaluator, "repobench"):
            msg_A = COMMUNICATION_CODE_MSG_TEMPLATE_A.format(instruction=CODE_INSTRUCTION, context=item["prompt_A"])
        elif hasattr(self.evaluator, "sasum"):
            msg_A = COMMUNICATION_SUMMARIZE_MSG_TEMPLATE_A.format(instruction=SUMMARIZE_INSTRUCTION, content_part_1=item["prompt_A"])
        else:
            msg_A = COMMUNICATION_QA_MSG_TEMPLATE_A.format(instruction=QA_INSTRUCTION, context=item["prompt_A"])
        input_ids_A = apply_chat_template(self.evaluator, self.tokenizer, msg_A, model_A, context=True)

        if hasattr(self.evaluator, "tmath"):
            msg_B = COMMUNICATION_MATH_MSG_TEMPLATE_B.format(instruction=MATH_INSTRUCTION, question=item["prompt_B"])
        elif hasattr(self.evaluator, "repobench"):
            msg_B = COMMUNICATION_CODE_MSG_TEMPLATE_B.format(instruction=CODE_INSTRUCTION, code_snippet=item["prompt_B"])
        elif hasattr(self.evaluator, "sasum"):
            msg_B = COMMUNICATION_SUMMARIZE_MSG_TEMPLATE_B.format(instruction=SUMMARIZE_INSTRUCTION, content_part_2=item["prompt_B"])
        else:
            msg_B = COMMUNICATION_QA_MSG_TEMPLATE_B.format(instruction=QA_INSTRUCTION, question=item["prompt_B"])
        input_ids_B = apply_chat_template(self.evaluator, self.tokenizer, msg_B, model_B)
        
        # truncate in the middle of the input
        input_ids_A, input_ids_B = self.truncate_input(input_ids_A, input_ids_B)

        return input_ids_A, input_ids_B

    def inference(self, model, cv, item):
        input_ids_A, input_ids_B = self.prepare_input_ids(item, cv.A, cv.B)

        out_A = model(
            input_ids=input_ids_A, 
            use_cache=True, 
            return_dict=True
        )
        out_A_past_key_values = out_A.past_key_values

        output = cv.generate(
            input_ids_B, 
            attention_mask=torch.ones_like(input_ids_B),
            out_A_past_key_values=out_A_past_key_values,
            **self.generate_args
        )[0]
        
        context_length = input_ids_B.shape[-1]
        response = self.get_response(output, context_length)
        return response

    def _test(self, model_A, cv, limit=None, calib_interval=10):
        i = 0
        recalibrated = False
        last_item = None
        while i < len(self.evaluator):
            if i % calib_interval == 0 and recalibrated:
                item = last_item
            else:
                item = next(self.evaluator)
                last_item = item
            if limit is not None and i >= limit:
                break
            if i % calib_interval == 0 and not recalibrated:
                cv.layers_list = list(range(0, cv.A_num_layers))
            response = self.inference(model_A, cv, item)

            if i % calib_interval == 0 and not recalibrated:
                recalibrated = True
                cv.calc_attn_weights_from_qk()
                self.layer_importance_total = calc_layer_importance(cv.B_attn_weights, model_A.name, self.layer_importance_total)
                self.cfg = get_top_layers(self.layer_importance_total, self.cfg)
                logging.info(f"Recalibrated layers_list: {np.sort(self.cfg.layers_list)}")
                cv.layers_list = self.cfg.layers_list
                self.layer_importance_total = defaultdict(list)
                i -= 1  # redo this sample with updated layers_list
            elif i % calib_interval == 0 and recalibrated:
                recalibrated = False
            i += 1
            
            self.evaluator.evaluate_item(item, response)
            
            result = self.evaluator.get_result()
            
        result = self.evaluator.get_result()
        return result
    
    @torch.no_grad()
    def test(self, model_A, cv, limit=None, calib_interval=10, no_wandb=False):
        result = self._test(model_A, cv, limit, calib_interval)
        if self.use_wandb and not no_wandb:
            wandb.log({f"{self.name}_result": result})
        logging.info(f"{self.name} result: {result:.4f}")
        return result
