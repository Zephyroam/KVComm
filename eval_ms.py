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

SKTLINE_QA_MSG_TEMPLATE = "Instruction: {instruction} Context: {context} Question: {question}"
SKTLINE_MATH_MSG_TEMPLATE = "Instruction: {instruction} Hint: {hint} Question: {question}"
SKTLINE_CODE_MSG_TEMPLATE = "Instruction: {instruction} Context: {context} Code Snippet: {code_snippet}"

BASELINE_QA_MSG_TEMPLATE = "Instruction: {instruction} Question: {question}"
BASELINE_MATH_MSG_TEMPLATE = "Instruction: {instruction} Question: {question}"
BASELINE_CODE_MSG_TEMPLATE = "Instruction: {instruction} Context: {context} Code Snippet: {code_snippet}"

COMMUNICATION_QA_MSG_TEMPLATE_A = "Instruction: {instruction} Context: {context}"
COMMUNICATION_QA_MSG_TEMPLATE_B = "Instruction: {instruction} Question: {question}"
COMMUNICATION_MATH_MSG_TEMPLATE_A = "Instruction: {instruction} Hint: {hint}"
COMMUNICATION_MATH_MSG_TEMPLATE_B = "Instruction: {instruction} Question: {question}"
COMMUNICATION_CODE_MSG_TEMPLATE_A = "Instruction: {instruction} Context: {context}"
COMMUNICATION_CODE_MSG_TEMPLATE_B = "Instruction: {instruction} Code Snippet: {question}"

SENDER_QA_INSTRUCTION = "Summarize the context passage in a concise way, as it will be used by another agent to answer the question."
SENDER_MATH_INSTRUCTION = "Summarize the hint in a concise way, as it will be used by another agent to answer the question."
SENDER_CODE_INSTRUCTION = "Summarize the code snippet in a concise way, as it will be used by another agent to complete the code."

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
        if not context and evaluator.name != "tipsheets":
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
    def __init__(self, evaluator, tokenizer, use_wandb, max_input_length):
        self.evaluator = evaluator
        self.tokenizer = tokenizer
        self.use_wandb = use_wandb
        self.max_input_length = max_input_length
        self.name = "communication"
        self.generate_args = {
            "max_new_tokens": self.evaluator.max_tokens,
            "temperature": 1.0,
            "num_beams": 1,
            "top_p": None,
            "top_k": None,
            "do_sample": False,
            "pad_token_id": self.tokenizer.eos_token_id
        }
        self.layer_importance_total = defaultdict(list)
    
    def truncate_input(self, input_ids_A1, input_ids_A2, input_ids_B):
        if input_ids_A1.shape[-1] + input_ids_A2.shape[-1] + input_ids_B.shape[-1] > self.max_input_length and self.evaluator.truncate_input:
            half = int((self.max_input_length - input_ids_B.shape[-1]) / 2)
            input_ids_A1 = input_ids_A1[:, :half]
            input_ids_A2 = input_ids_A2[:, -half:]
        return input_ids_A1, input_ids_A2, input_ids_B

    def get_response(self, output, context_length, truncate_response=True):
        if truncate_response:
            response = self.tokenizer.decode(output[context_length:], skip_special_tokens=True)
        else:
            response = self.tokenizer.decode(output, skip_special_tokens=True)
        return response

    def prepare_input_ids(self, item, model_A1, model_A2, model_B):
        def _get_msg_A(prompt_A):
            if hasattr(self.evaluator, "tmath"):
                msg_A = COMMUNICATION_MATH_MSG_TEMPLATE_A.format(instruction=MATH_INSTRUCTION, hint=prompt_A)
            elif hasattr(self.evaluator, "repobench"):
                msg_A = COMMUNICATION_CODE_MSG_TEMPLATE_A.format(instruction=CODE_INSTRUCTION, context=prompt_A)
            else:
                msg_A = COMMUNICATION_QA_MSG_TEMPLATE_A.format(instruction=QA_INSTRUCTION, context=prompt_A)
            return msg_A
        input_ids_A1 = apply_chat_template(self.evaluator, self.tokenizer, _get_msg_A(item["prompt_A1"]), model_A1, context=True)
        input_ids_A2 = apply_chat_template(self.evaluator, self.tokenizer, _get_msg_A(item["prompt_A2"]), model_A2, context=True)

        if hasattr(self.evaluator, "tmath"):
            msg_B = COMMUNICATION_MATH_MSG_TEMPLATE_B.format(instruction=MATH_INSTRUCTION, question=item["prompt_B"])
        elif hasattr(self.evaluator, "repobench"):
            msg_B = COMMUNICATION_CODE_MSG_TEMPLATE_B.format(instruction=CODE_INSTRUCTION, code_snippet=item["prompt_B"])
        else:
            msg_B = COMMUNICATION_QA_MSG_TEMPLATE_B.format(instruction=QA_INSTRUCTION, question=item["prompt_B"])
        input_ids_B = apply_chat_template(self.evaluator, self.tokenizer, msg_B, model_B)
        
        # truncate in the middle of the input
        input_ids_A1, input_ids_A2, input_ids_B = self.truncate_input(input_ids_A1, input_ids_A2, input_ids_B)

        return input_ids_A1, input_ids_A2, input_ids_B

    def inference(self, model_A1, model_A2, cv, item):
        input_ids_A1, input_ids_A2, input_ids_B = self.prepare_input_ids(item, cv.A1, cv.A2, cv.B)

        out_A1 = model_A1(
            input_ids=input_ids_A1, 
            use_cache=True, 
            return_dict=True
        )
        out_A1_past_key_values = out_A1.past_key_values
        out_A2 = model_A2(
            input_ids=input_ids_A2, 
            use_cache=True, 
            return_dict=True
        )
        out_A2_past_key_values = out_A2.past_key_values

        output = cv.generate(
            input_ids_B, 
            attention_mask=torch.ones_like(input_ids_B),
            out_A1_past_key_values=out_A1_past_key_values,
            out_A2_past_key_values=out_A2_past_key_values,
            **self.generate_args
        )[0]
        
        context_length = input_ids_B.shape[-1]
        response = self.get_response(output, context_length)
        return response

    def _test(self, model_A1, model_A2, cv, limit=None, do_calc_layer_importance=False):
        progress_bar = tqdm(self.evaluator, desc=f"{self.name} result: 0.0000", disable=do_calc_layer_importance)

        for i, item in enumerate(progress_bar):
            if limit is not None and i >= limit:
                break
            response = self.inference(model_A1, model_A2, cv, item)

            if do_calc_layer_importance:
                cv.calc_attn_weights_from_qk()
                self.layer_importance_total = calc_layer_importance(cv.B_attn_weights, None, self.layer_importance_total)
            
            self.evaluator.evaluate_item(item, response)
            
            result = self.evaluator.get_result()
            progress_bar.set_description(f"{self.name} result: {result:.4f}")
            
        result = self.evaluator.get_result()
        return result
    
    @torch.no_grad()
    def test(self, model_A1, model_A2, cv, limit=None, do_calc_layer_importance=False, no_wandb=False):
        tic = time.time()
        result = self._test(model_A1, model_A2, cv, limit, do_calc_layer_importance)
        toc = time.time()
        time_used = toc - tic
        if self.use_wandb and not no_wandb and not do_calc_layer_importance:
            wandb.log({f"{self.name}_result": result, f"{self.name}_time": time_used})
        logging.info(f"{self.name} result: {result:.4f}, {self.name} time: {time_used:.2f}s")
        return result


REFINE_TMPL = "{prompt}\nYour previous answer:\n{self_answer}\nAnother agent's answer (for your consideration):\n{others_A1}\nAnother agent's answer (for your consideration):\n{others_A2}\nIf needed, revise your answer. Your new answer is:"


class NLDEvaluator(CommunicationEvaluator):
    def __init__(self, evaluator, tokenizer, use_wandb, max_input_length, max_tokens_A_model_phase1, sender_aware=False):
        super().__init__(evaluator, tokenizer, use_wandb, max_input_length)
        self.name = "nld"
        self.max_tokens_phase_1 = max_tokens_A_model_phase1
        self.sender_aware = sender_aware

    def prepare_input_ids(self, item, model_A1, model_A2, model_B):
        def _get_msg_A(prompt_A):
            if self.sender_aware:
                if hasattr(self.evaluator, "tmath"):
                    msg_A = COMMUNICATION_MATH_MSG_TEMPLATE_A.format(instruction=SENDER_MATH_INSTRUCTION, hint=prompt_A)
                elif hasattr(self.evaluator, "repobench"):
                    msg_A = COMMUNICATION_CODE_MSG_TEMPLATE_A.format(instruction=SENDER_CODE_INSTRUCTION, context=prompt_A)
                else:
                    msg_A = COMMUNICATION_QA_MSG_TEMPLATE_A.format(instruction=SENDER_QA_INSTRUCTION, context=prompt_A)
            else:
                if hasattr(self.evaluator, "tmath"):
                    msg_A = COMMUNICATION_MATH_MSG_TEMPLATE_A.format(instruction=MATH_INSTRUCTION, hint=prompt_A)
                elif hasattr(self.evaluator, "repobench"):
                    msg_A = COMMUNICATION_CODE_MSG_TEMPLATE_A.format(instruction=CODE_INSTRUCTION, context=prompt_A)
                else:
                    msg_A = COMMUNICATION_QA_MSG_TEMPLATE_A.format(instruction=QA_INSTRUCTION, context=prompt_A)
            return msg_A
        input_ids_A1 = apply_chat_template(self.evaluator, self.tokenizer, _get_msg_A(item["prompt_A1"]), model_A1)
        input_ids_A2 = apply_chat_template(self.evaluator, self.tokenizer, _get_msg_A(item["prompt_A2"]), model_A2)

        if hasattr(self.evaluator, "tmath"):
            msg_B = COMMUNICATION_MATH_MSG_TEMPLATE_B.format(instruction=MATH_INSTRUCTION, question=item["prompt_B"])
        elif hasattr(self.evaluator, "repobench"):
            msg_B = COMMUNICATION_CODE_MSG_TEMPLATE_B.format(instruction=CODE_INSTRUCTION, code_snippet=item["prompt_B"])
        else:
            msg_B = COMMUNICATION_QA_MSG_TEMPLATE_B.format(instruction=QA_INSTRUCTION, question=item["prompt_B"])
        input_ids_B = apply_chat_template(self.evaluator, self.tokenizer, msg_B, model_B)
        
        # truncate in the middle of the input
        input_ids_A1, input_ids_A2, input_ids_B = self.truncate_input(input_ids_A1, input_ids_A2, input_ids_B)

        return input_ids_A1, input_ids_A2, input_ids_B, msg_B

    def truncate_input_nld(self, input_ids):
        if input_ids.shape[-1] > self.max_input_length and self.evaluator.truncate_input:
            half = int(self.max_input_length / 2)
            input_ids = torch.cat([input_ids[:, :half], input_ids[:, -half:]], dim=-1)
        return input_ids

    def prepare_input_ids_nld(self, prompt: str, self_answer: str, others_A1: str, others_A2: str, model):
        msg = REFINE_TMPL.format(prompt=prompt, self_answer=self_answer, others_A1=others_A1, others_A2=others_A2)
        input_ids = apply_chat_template(self.evaluator, self.tokenizer, msg, model)
        
        # truncate in the middle of the input
        input_ids = self.truncate_input_nld(input_ids)
        return input_ids

    def inference(self, model_A1, model_A2, model_B, item):
        input_ids_A1, input_ids_A2, input_ids_B, msg_B = self.prepare_input_ids(item, model_A1, model_A2, model_B)
        # overwrite max_new_tokens for model A and model B for phase 1
        self.generate_args["max_new_tokens"] = self.max_tokens_phase_1

        output = model_A1.generate(
            input_ids_A1, 
            attention_mask=torch.ones_like(input_ids_A1),
            **self.generate_args,
        )[0]
        context_length = input_ids_A1.shape[-1]
        initial_answer_A1 = self.get_response(output, context_length)

        output = model_A2.generate(
            input_ids_A2, 
            attention_mask=torch.ones_like(input_ids_A2),
            **self.generate_args,
        )[0]
        context_length = input_ids_A2.shape[-1]
        initial_answer_A2 = self.get_response(output, context_length)
        
        output = model_B.generate(
            input_ids_B, 
            attention_mask=torch.ones_like(input_ids_B),
            **self.generate_args
        )[0]
        context_length = input_ids_B.shape[-1]
        initial_answer_B = self.get_response(output, context_length)

        # restore generation for new tokens
        self.generate_args["max_new_tokens"] = self.evaluator.max_tokens

        input_ids = self.prepare_input_ids_nld(msg_B, initial_answer_B, initial_answer_A1, initial_answer_A2, model_B)
        output = model_B.generate(
            input_ids, 
            attention_mask=torch.ones_like(input_ids),
            **self.generate_args
        )[0]
        context_length = input_ids.shape[-1]
        response = self.get_response(output, context_length)
        return response

    def _test(self, model_A1, model_A2, model_B, limit=None):
        progress_bar = tqdm(self.evaluator, desc=f"{self.name} result: 0.0000")

        for i, item in enumerate(progress_bar):
            if limit is not None and i >= limit:
                break
            response = self.inference(model_A1, model_A2, model_B, item)
            
            self.evaluator.evaluate_item(item, response)
            
            result = self.evaluator.get_result()
            progress_bar.set_description(f"{self.name} result: {result:.4f}")
            
        result = self.evaluator.get_result()
        return result
    
    @torch.no_grad()
    def test(self, model_A1, model_A2, model_B, limit=None):
        tic = time.time()
        result = self._test(model_A1, model_A2, model_B, limit)
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

    def prepare_inputs_embeds_cipher(self, prompt: str, cipher_embeds_self: torch.Tensor, cipher_embeds_others_A1: torch.Tensor, cipher_embeds_others_A2: torch.Tensor, model):
        msg = REFINE_TMPL.format(prompt=prompt, self_answer="<SELF_ANS>", others_A1="<OTHERS_ANS>", others_A2="<OTHERS_ANS>")
        input_ids = apply_chat_template(self.evaluator, self.tokenizer, msg, model.model)[0]

        sentinel_positions = (input_ids == model.SELF_ID).nonzero(as_tuple=False), (input_ids == model.OTHERS_ID).nonzero(as_tuple=False)
        self_pos = sentinel_positions[0][0].item()
        others_A1_pos = sentinel_positions[1][0].item()
        others_A2_pos = sentinel_positions[1][1].item()
        if not (0 <= self_pos < others_A1_pos < others_A2_pos < input_ids.shape[-1]):
            raise RuntimeError("Unexpected sentinel positions")

        prefix_ids  = input_ids[:self_pos]
        middle1_ids  = input_ids[self_pos+1:others_A1_pos]
        middle2_ids  = input_ids[others_A1_pos+1:others_A2_pos]
        suffix_ids  = input_ids[others_A2_pos+1:]

        prefix_emb = F.embedding(prefix_ids, model.embed_weight)
        middle1_emb = F.embedding(middle1_ids, model.embed_weight)
        middle2_emb = F.embedding(middle2_ids, model.embed_weight)
        suffix_emb = F.embedding(suffix_ids, model.embed_weight)

        cipher_embeds_self = cipher_embeds_self.squeeze(0)
        cipher_embeds_others_A1 = cipher_embeds_others_A1.squeeze(0)
        cipher_embeds_others_A2 = cipher_embeds_others_A2.squeeze(0)
        
        inputs_embeds = torch.cat(
            [prefix_emb, cipher_embeds_self, middle1_emb, cipher_embeds_others_A1, middle2_emb, cipher_embeds_others_A2, suffix_emb],
            dim=0
        ).unsqueeze(0)  
        # truncate in the middle of the input
        assert inputs_embeds.shape[1] <= self.max_input_length, "Input length is too long"
        return inputs_embeds

    def inference(self, model_A1, model_A2, model_B, item):
        input_ids_A1, input_ids_A2, input_ids_B, msg_B = self.prepare_input_ids(item, model_A1.model, model_A2.model, model_B.model)
        # overwrite max_new_tokens for model A and model B for phase 1
        self.generate_args["max_new_tokens"] = self.max_tokens_phase_1

        cipher_embeds_A1 = model_A1.cipher_generate(
            input_ids=input_ids_A1, 
            attention_mask=torch.ones_like(input_ids_A1),
            **self.generate_args,
        )
        cipher_embeds_A2 = model_A2.cipher_generate(
            input_ids=input_ids_A2, 
            attention_mask=torch.ones_like(input_ids_A2),
            **self.generate_args,
        )

        cipher_embeds_B = model_B.cipher_generate(
            input_ids=input_ids_B, 
            attention_mask=torch.ones_like(input_ids_B),
            **self.generate_args
        )

        # restore generation for new tokens
        self.generate_args["max_new_tokens"] = self.evaluator.max_tokens

        inputs_embeds = self.prepare_inputs_embeds_cipher(msg_B, cipher_embeds_B, cipher_embeds_A1, cipher_embeds_A2, model_B)
        output = model_B.generate(
            inputs_embeds=inputs_embeds, 
            attention_mask=torch.ones_like(inputs_embeds[..., 0]),
            **self.generate_args
        )[0]
        response = self.get_response(output, None, truncate_response=False)
        return response
