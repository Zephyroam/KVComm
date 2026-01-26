import torch
import torch.nn.functional as F
import logging
from tqdm import tqdm
import wandb
from layer_importance import calc_layer_importance
from collections import defaultdict
import time

QA_INSTRUCTION = "Directly answer the question based on the context passage, no explanation is needed."
MATH_INSTRUCTION = "Answer the math problem step by step."
CODE_INSTRUCTION = "Complete ONLY THE NEXT LINE of the code snippet based on the context."
SUMMARIZE_INSTRUCTION = "Summarize the following content concisely with one sentence."

SKTLINE_QA_MSG_TEMPLATE = "Instruction: {instruction} Context: {context} Question: {question}"
SKTLINE_MATH_MSG_TEMPLATE = "Instruction: {instruction} Hint: {hint} Question: {question}"
SKTLINE_CODE_MSG_TEMPLATE = "Instruction: {instruction} Context: {context} Code Snippet: {code_snippet}"
SKTLINE_SUMMARIZE_MSG_TEMPLATE = "Instruction: {instruction} Content part 1: {content_part_1} Content part 2: {content_part_2}"

BASELINE_QA_MSG_TEMPLATE = "Instruction: {instruction} Question: {question}"
BASELINE_MATH_MSG_TEMPLATE = "Instruction: {instruction} Question: {question}"
BASELINE_CODE_MSG_TEMPLATE = "Instruction: {instruction} Code Snippet: {code_snippet}"
BASELINE_SUMMARIZE_MSG_TEMPLATE = "Instruction: {instruction} Content: {content_part_2}"

COMMUNICATION_QA_MSG_TEMPLATE_A = "Instruction: {instruction} Context: {context}"
COMMUNICATION_QA_MSG_TEMPLATE_B = "Instruction: {instruction} Question: {question}"
COMMUNICATION_MATH_MSG_TEMPLATE_A = "Instruction: {instruction} Hint: {hint}"
COMMUNICATION_MATH_MSG_TEMPLATE_B = "Instruction: {instruction} Question: {question}"
COMMUNICATION_CODE_MSG_TEMPLATE_A = "Instruction: {instruction} Context: {context}"
COMMUNICATION_CODE_MSG_TEMPLATE_B = "Instruction: {instruction} Code Snippet: {code_snippet}"
COMMUNICATION_SUMMARIZE_MSG_TEMPLATE_A = "Instruction: {instruction} Content part 1: {content_part_1}"
COMMUNICATION_SUMMARIZE_MSG_TEMPLATE_B = "Instruction: {instruction} Content part 2: {content_part_2}"

SENDER_QA_INSTRUCTION = "Summarize the context passage in a concise way, as it will be used by another agent to answer the question."
SENDER_MATH_INSTRUCTION = "Summarize the hint in a concise way, as it will be used by another agent to answer the question."
SENDER_CODE_INSTRUCTION = "Summarize the code snippet in a concise way, as it will be used by another agent to complete the code."
SENDER_SUMMARIZE_INSTRUCTION = "Summarize the content in a concise way, as it will be used by another agent to understand the content."

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

class SkylineEvaluator:
    def __init__(self, evaluator, tokenizer, use_wandb, max_input_length):
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
    
    def truncate_input(self, input_ids):
        if input_ids.shape[-1] > self.max_input_length and self.evaluator.truncate_input:
            half = int(self.max_input_length / 2)
            input_ids = torch.cat([input_ids[:, :half], input_ids[:, -half:]], dim=-1)
        return input_ids

    def prepare_input_ids(self, item, model):
        if hasattr(self.evaluator, "tmath"):
            msg = SKTLINE_MATH_MSG_TEMPLATE.format(instruction=MATH_INSTRUCTION, hint=item["prompt_A"], question=item["prompt_B"])
        elif hasattr(self.evaluator, "repobench"):
            msg = SKTLINE_CODE_MSG_TEMPLATE.format(instruction=CODE_INSTRUCTION, context=item["prompt_A"], code_snippet=item["prompt_B"])
        elif hasattr(self.evaluator, "sasum"):
            msg = SKTLINE_SUMMARIZE_MSG_TEMPLATE.format(instruction=SUMMARIZE_INSTRUCTION, content_part_1=item["prompt_A"], content_part_2=item["prompt_B"])
        else:
            msg = SKTLINE_QA_MSG_TEMPLATE.format(instruction=QA_INSTRUCTION, context=item["prompt_A"], question=item["prompt_B"])
        input_ids = apply_chat_template(self.evaluator, self.tokenizer, msg, model)
        
        # truncate in the middle of the input
        input_ids = self.truncate_input(input_ids)
        return input_ids

    def get_response(self, output, context_length, truncate_response=True):
        if truncate_response:
            response = self.tokenizer.decode(output[context_length:], skip_special_tokens=True)
        else:
            response = self.tokenizer.decode(output, skip_special_tokens=True)
        return response

    def inference(self, model, item):
        input_ids = self.prepare_input_ids(item, model)

        output = model.generate(
            input_ids, 
            attention_mask=torch.ones_like(input_ids),
            **self.generate_args
        )[0]
        
        context_length = input_ids.shape[-1]
        response = self.get_response(output, context_length)
        return response

    def _test(self, model, limit=None):
        progress_bar = tqdm(self.evaluator, desc=f"{self.name} result: 0.0000")
            
        for i, item in enumerate(progress_bar):
            if limit is not None and i >= limit:
                break
            response = self.inference(model, item)
            
            self.evaluator.evaluate_item(item, response)
            
            result = self.evaluator.get_result()
            progress_bar.set_description(f"{self.name} result: {result:.4f}")
            
        result = self.evaluator.get_result()
        return result
    
    @torch.no_grad()
    def test(self, model_A, model_B, limit=None):
        tic = time.time()
        result_A = self._test(model_A, limit)
        toc = time.time()
        time_A = toc - tic
        tic = time.time()
        result_B = self._test(model_B, limit)
        toc = time.time()
        time_B = toc - tic
        
        if self.use_wandb:
            wandb.log({f"{self.name}_result_A": result_A, f"{self.name}_result_B": result_B, f"{self.name}_time_A": time_A, f"{self.name}_time_B": time_B})
        logging.info(f"{self.name} result A: {result_A:.4f}, {self.name} result B: {result_B:.4f}, {self.name} time A: {time_A:.2f}s, {self.name} time B: {time_B:.2f}s")
        return result_A, result_B

class BaselineEvaluator(SkylineEvaluator):
    def __init__(self, evaluator, tokenizer, use_wandb, max_input_length):
        super().__init__(evaluator, tokenizer, use_wandb, max_input_length)
        self.name = "baseline"
        
    def prepare_input_ids(self, item, model):
        if hasattr(self.evaluator, "tmath"):
            msg = BASELINE_MATH_MSG_TEMPLATE.format(instruction=MATH_INSTRUCTION, question=item["prompt_B"])
        elif hasattr(self.evaluator, "repobench"):
            msg = BASELINE_CODE_MSG_TEMPLATE.format(instruction=CODE_INSTRUCTION, code_snippet=item["prompt_B"])
        elif hasattr(self.evaluator, "sasum"):
            msg = BASELINE_SUMMARIZE_MSG_TEMPLATE.format(instruction=SUMMARIZE_INSTRUCTION, content_part_2=item["prompt_B"])
        else:
            msg = BASELINE_QA_MSG_TEMPLATE.format(instruction=QA_INSTRUCTION, question=item["prompt_B"])
        input_ids = apply_chat_template(self.evaluator, self.tokenizer, msg, model)
        
        # truncate in the middle of the input
        input_ids = self.truncate_input(input_ids)
        return input_ids

class CommunicationEvaluator(SkylineEvaluator):
    def __init__(self, evaluator, tokenizer, use_wandb, max_input_length):
        super().__init__(evaluator, tokenizer, use_wandb, max_input_length)
        self.name = "communication"
        self.layer_importance_total = defaultdict(list)
    
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

    def _test(self, model_A, cv, limit=None, do_calc_layer_importance=False):
        progress_bar = tqdm(self.evaluator, desc=f"{self.name} result: 0.0000", disable=do_calc_layer_importance)

        for i, item in enumerate(progress_bar):
            if limit is not None and i >= limit:
                break
            response = self.inference(model_A, cv, item)

            if do_calc_layer_importance:
                cv.calc_attn_weights_from_qk()
                self.layer_importance_total = calc_layer_importance(cv.B_attn_weights, model_A.name, self.layer_importance_total)
            
            self.evaluator.evaluate_item(item, response)
            
            result = self.evaluator.get_result()
            progress_bar.set_description(f"{self.name} result: {result:.4f}")
            
        result = self.evaluator.get_result()
        return result
    
    @torch.no_grad()
    def test(self, model_A, cv, limit=None, do_calc_layer_importance=False, no_wandb=False):
        tic = time.time()
        result = self._test(model_A, cv, limit, do_calc_layer_importance)
        toc = time.time()
        time_used = toc - tic
        if self.use_wandb and not no_wandb and not do_calc_layer_importance:
            wandb.log({f"{self.name}_result": result, f"{self.name}_time": time_used})
        logging.info(f"{self.name} result: {result:.4f}, {self.name} time: {time_used:.2f}s")
        return result

class ACEvaluator(CommunicationEvaluator):
    def __init__(self, evaluator, tokenizer, use_wandb, max_input_length):
        super().__init__(evaluator, tokenizer, use_wandb, max_input_length)
        self.name = "ac"

    def inference(self, model, ac, item):
        input_ids_A, input_ids_B = self.prepare_input_ids(item, ac.A, ac.B)

        out_A = model(
            input_ids=input_ids_A, 
            use_cache=True, 
            output_hidden_states=True, 
            return_dict=True
        )

        output = ac.generate(
            input_ids_B, 
            attention_mask=torch.ones_like(input_ids_B),
            h_A=out_A.hidden_states,
            **self.generate_args
        )[0]
        
        context_length = input_ids_B.shape[-1]
        response = self.get_response(output, context_length)
        return response

    def _test(self, model_A, ac, limit=None):
        progress_bar = tqdm(self.evaluator, desc=f"{self.name} result: 0.0000")

        for i, item in enumerate(progress_bar):
            if limit is not None and i >= limit:
                break
            response = self.inference(model_A, ac, item)

            self.evaluator.evaluate_item(item, response)
            
            result = self.evaluator.get_result()
            progress_bar.set_description(f"{self.name} result: {result:.4f}")
            
        result = self.evaluator.get_result()
        return result
    
    @torch.no_grad()
    def test(self, model_A, ac, limit=None):
        tic = time.time()
        result = self._test(model_A, ac, limit)
        toc = time.time()
        time_used = toc - tic
        if self.use_wandb:
            wandb.log({f"{self.name}_result": result, f"{self.name}_time": time_used})
        logging.info(f"{self.name} result: {result:.4f}, {self.name} time: {time_used:.2f}s")
        return result

REFINE_TMPL = "{prompt}\nYour previous answer:\n{self_answer}\nOther agents' answers (for your consideration):\n{others}\nIf needed, revise your answer. Your new answer is:"


class NLDEvaluator(CommunicationEvaluator):
    def __init__(self, evaluator, tokenizer, use_wandb, max_input_length, max_tokens_A_model_phase1, sender_aware=False):
        super().__init__(evaluator, tokenizer, use_wandb, max_input_length)
        self.name = "nld"
        self.max_tokens_phase_1 = max_tokens_A_model_phase1
        self.sender_aware = sender_aware

    def prepare_input_ids(self, item, model_A, model_B):
        if self.sender_aware:
            if hasattr(self.evaluator, "tmath"):
                msg_A = COMMUNICATION_MATH_MSG_TEMPLATE_A.format(instruction=SENDER_MATH_INSTRUCTION, hint=item["prompt_A"])
            elif hasattr(self.evaluator, "repobench"):
                msg_A = COMMUNICATION_CODE_MSG_TEMPLATE_A.format(instruction=SENDER_CODE_INSTRUCTION, context=item["prompt_A"])
            elif hasattr(self.evaluator, "sasum"):
                msg_A = COMMUNICATION_SUMMARIZE_MSG_TEMPLATE_A.format(instruction=SENDER_SUMMARIZE_INSTRUCTION, content_part_1=item["prompt_A"])
            else:
                msg_A = COMMUNICATION_QA_MSG_TEMPLATE_A.format(instruction=SENDER_QA_INSTRUCTION, context=item["prompt_A"])
        else:
            if hasattr(self.evaluator, "tmath"):
                msg_A = COMMUNICATION_MATH_MSG_TEMPLATE_A.format(instruction=MATH_INSTRUCTION, hint=item["prompt_A"])
            elif hasattr(self.evaluator, "repobench"):
                msg_A = COMMUNICATION_CODE_MSG_TEMPLATE_A.format(instruction=CODE_INSTRUCTION, context=item["prompt_A"])
            elif hasattr(self.evaluator, "sasum"):
                msg_A = COMMUNICATION_SUMMARIZE_MSG_TEMPLATE_A.format(instruction=SUMMARIZE_INSTRUCTION, content_part_1=item["prompt_A"])
            else:
                msg_A = COMMUNICATION_QA_MSG_TEMPLATE_A.format(instruction=QA_INSTRUCTION, context=item["prompt_A"])
        input_ids_A = apply_chat_template(self.evaluator, self.tokenizer, msg_A, model_A)

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

        return input_ids_A, input_ids_B, msg_B

    def truncate_input_nld(self, input_ids):
        if input_ids.shape[-1] > self.max_input_length and self.evaluator.truncate_input:
            half = int(self.max_input_length / 2)
            input_ids = torch.cat([input_ids[:, :half], input_ids[:, -half:]], dim=-1)
        return input_ids

    def prepare_input_ids_nld(self, prompt: str, self_answer: str, others: str, model):
        msg = REFINE_TMPL.format(prompt=prompt, self_answer=self_answer, others=others)
        input_ids = apply_chat_template(self.evaluator, self.tokenizer, msg, model)
        
        # truncate in the middle of the input
        input_ids = self.truncate_input_nld(input_ids)
        return input_ids

    def inference(self, model_A, model_B, item):
        input_ids_A, input_ids_B, msg_B = self.prepare_input_ids(item, model_A, model_B)
        # overwrite max_new_tokens for model A and model B for phase 1
        self.generate_args["max_new_tokens"] = self.max_tokens_phase_1

        output = model_A.generate(
            input_ids_A, 
            attention_mask=torch.ones_like(input_ids_A),
            **self.generate_args,
        )[0]

        context_length = input_ids_A.shape[-1]
        initial_answer_A = self.get_response(output, context_length)

        output = model_B.generate(
            input_ids_B, 
            attention_mask=torch.ones_like(input_ids_B),
            **self.generate_args
        )[0]
        context_length = input_ids_B.shape[-1]
        initial_answer_B = self.get_response(output, context_length)

        # restore generation for new tokens
        self.generate_args["max_new_tokens"] = self.evaluator.max_tokens

        input_ids = self.prepare_input_ids_nld(msg_B, initial_answer_B, initial_answer_A, model_B)
        output = model_B.generate(
            input_ids, 
            attention_mask=torch.ones_like(input_ids),
            **self.generate_args
        )[0]
        context_length = input_ids.shape[-1]
        response = self.get_response(output, context_length)
        return response

    def _test(self, model_A, model_B, limit=None):
        progress_bar = tqdm(self.evaluator, desc=f"{self.name} result: 0.0000")

        for i, item in enumerate(progress_bar):
            if limit is not None and i >= limit:
                break
            try:
                response = self.inference(model_A, model_B, item)
            except Exception as e:
                logging.error(f"Error during inference: {e}")
                continue
            
            self.evaluator.evaluate_item(item, response)
            
            result = self.evaluator.get_result()
            progress_bar.set_description(f"{self.name} result: {result:.4f}")
            
        result = self.evaluator.get_result()
        return result
    
    @torch.no_grad()
    def test(self, model_A, model_B, limit=None):
        tic = time.time()
        result = self._test(model_A, model_B, limit)
        toc = time.time()
        time_used = toc - tic
        if self.use_wandb:
            wandb.log({f"{self.name}_result": result, f"{self.name}_time": time_used})
        logging.info(f"{self.name} result: {result:.4f}, {self.name} time: {time_used:.2f}s")
        return result



class CipherEvaluator(NLDEvaluator):
    def __init__(self, evaluator, tokenizer, use_wandb, max_input_length, max_tokens_A_model_phase1, sender_aware=False):
        super().__init__(evaluator, tokenizer, use_wandb, max_input_length, max_tokens_A_model_phase1, sender_aware)
        self.name = "cipher"
        self.max_tokens_phase_1 = max_tokens_A_model_phase1

    def prepare_inputs_embeds_cipher(self, prompt: str, cipher_embeds_self: torch.Tensor, cipher_embeds_others: torch.Tensor, model):
        msg = REFINE_TMPL.format(prompt=prompt, self_answer="<SELF_ANS>", others="<OTHERS_ANS>")
        input_ids = apply_chat_template(self.evaluator, self.tokenizer, msg, model.model)[0]

        sentinel_positions = (input_ids == model.SELF_ID).nonzero(as_tuple=False), (input_ids == model.OTHERS_ID).nonzero(as_tuple=False)
        self_pos = sentinel_positions[0][0].item()
        others_pos = sentinel_positions[1][0].item()
        if not (0 <= self_pos < others_pos < input_ids.numel()):
            raise RuntimeError("Unexpected sentinel positions")

        prefix_ids  = input_ids[:self_pos]
        middle_ids  = input_ids[self_pos+1:others_pos]
        suffix_ids  = input_ids[others_pos+1:]

        prefix_emb = F.embedding(prefix_ids, model.embed_weight)
        middle_emb = F.embedding(middle_ids, model.embed_weight)
        suffix_emb = F.embedding(suffix_ids, model.embed_weight)

        cipher_embeds_self = cipher_embeds_self.squeeze(0)
        cipher_embeds_others = cipher_embeds_others.squeeze(0)
        
        inputs_embeds = torch.cat(
            [prefix_emb, cipher_embeds_self, middle_emb, cipher_embeds_others, suffix_emb],
            dim=0
        ).unsqueeze(0)  
        # truncate in the middle of the input
        assert inputs_embeds.shape[1] <= self.max_input_length, "Input length is too long"
        return inputs_embeds

    def inference(self, model_A, model_B, item):
        input_ids_A, input_ids_B, msg_B = self.prepare_input_ids(item, model_A.model, model_B.model)
        # overwrite max_new_tokens for model A and model B for phase 1
        self.generate_args["max_new_tokens"] = self.max_tokens_phase_1

        cipher_embeds_A = model_A.cipher_generate(
            input_ids=input_ids_A, 
            attention_mask=torch.ones_like(input_ids_A),
            **self.generate_args,
        )

        cipher_embeds_B = model_B.cipher_generate(
            input_ids=input_ids_B, 
            attention_mask=torch.ones_like(input_ids_B),
            **self.generate_args
        )

        # restore generation for new tokens
        self.generate_args["max_new_tokens"] = self.evaluator.max_tokens

        inputs_embeds = self.prepare_inputs_embeds_cipher(msg_B, cipher_embeds_B, cipher_embeds_A, model_B)
        output = model_B.generate(
            inputs_embeds=inputs_embeds, 
            attention_mask=torch.ones_like(inputs_embeds[..., 0]),
            **self.generate_args
        )[0]
        response = self.get_response(output, None, truncate_response=False)
        return response
