import torch
import math
import logging
import time
import re
import numpy as np


class SafeDecodingTiny:
    def __init__(self, model, tokenizer, expert_model,expert_tokenizer,
                 verbose=False,
                 div_metric: str = "l1_norm",
                 gate_method: str = "lif",
                 top_k_merge: int = 64, temperature: float = 1.0,
                 # LIF的超参
                 v_thres: float = 1,
                 v_reset: float = 0.0,
                 tau: float = 8.0,
                 # 略大 下降缓慢
                 soft_reset: float = 0.6,

                 # SafeDecoding strength (alpha in logit interpolation)
                 sd_alpha: float = 0.5,

                 # Apply SafeDecoding only to the first m generated tokens
                 sd_m: int = 2,

                 # 是否使用动态alpha
                 if_dynamic_alpha: bool = False,
                 gamma: float = 2.0,
                 **kwargs):

        self.model = model
        self.tokenizer = tokenizer
        self.expert_model = expert_model
        self.expert_tokenizer = expert_tokenizer
        self.verbose = verbose
        self.do_sample = False
        self.top_p = None
        # divergence-based parameters
        self.div_metric = div_metric.lower()
        self.gate_method = gate_method.lower()

        self.top_k_merge = top_k_merge
        self.temperature = temperature

        # LIF neuron state & params
        self.v_thres = float(v_thres)
        self.v_reset = float(v_reset)
        self.tau = float(tau)
        self.v = float(v_reset)
        self.soft_reset = float(soft_reset)

        self.num_spike = 0

        # SafeDecoding interpolation factor
        self.sd_alpha = float(sd_alpha)
        self.sd_alpha = 0.5

        # Apply SafeDecoding only to the first m generated tokens
        self.sd_m = int(sd_m)

        # 是否使用动态alpha
        self.if_dynamic_alpha = False
        self.gamma = gamma


        # 更新的早停机制 使用buffer
        self.early_stop_buffer = []

        logging.info("DistributionSafeDecoding with LIF gate initialized.")

        print("-------------------------------")
        print("[Simple Expert Blend Decoding]")
        print("Blend base + expert only for the first m generated tokens.")
        print(f"Expert weight (alpha): {self.sd_alpha}")
        print(f"SafeDecoding steps (m): {self.sd_m}")
        print("-------------------------------")


    def _safe_adjust_topk(self, p_base: torch.Tensor, p_expert: torch.Tensor, top_k: int = None, current=None):
        """Simplified: always blend base & expert on a union top-k candidate set.

        This version removes any gating / dynamic-alpha logic.
        For tokens in the candidate set:
            score = 0.5 * p_base + 0.5 * p_expert
        Elsewhere, keep base probability.

        Returns best token index and the adjusted score tensor.
        """
        p_expert = p_expert.to(p_base.device)
        if top_k is None:
            top_k = self.top_k_merge

        kb = min(top_k, p_base.size(0))
        ke = min(top_k, p_expert.size(0))
        topb = torch.topk(p_base, k=kb).indices.tolist()
        tope = torch.topk(p_expert, k=ke).indices.tolist()
        cand = list(set(topb) | set(tope))

        score = p_base.clone()
        alpha = 0.5  # fixed expert weight
        if cand:
            score[cand] = (1.0 - alpha) * p_base[cand] + alpha * p_expert[cand]

        best_idx = int(torch.argmax(score).item())
        return best_idx, score

    def _select_token(self, p_base: torch.Tensor, p_expert: torch.Tensor, top_k: int = None):
        """Simplified: no divergence/gate; always use expert to adjust base with weight 0.5."""
        p_expert = p_expert.to(p_base.device)
        idx, _ = self._safe_adjust_topk(p_base, p_expert, top_k=top_k, current=None)
        # Keep return signature for compatibility with existing call sites
        return idx, False, 0.0, 0.0

    def reset(self,):
        self.v = float(self.v_reset)
        self.num_spike = 0

        self.adam_t = 0
        self.adam_m = 0.0
        self.adam_s = 0.0
        return




    # 采用SSD
    def safedecoding_speculative(self, inputs, expert_inputs=None, gen_config=None,return_generated_sequence=False, **kwargs):
        if gen_config is None:
            gen_config = self.model.generation_config

        # Override the generation config for our decoding
        self.do_sample = gen_config.do_sample
        self.top_p = gen_config.top_p
        # if self.verbose:
        #     logging.info(f"Generation config: {gen_config}")

        inputs = {k:v.cuda(self.model.device) for k,v in inputs.items()}
        generated_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        input_len = inputs['input_ids'].shape[1]
        batch_size = inputs['input_ids'].shape[0]
        assert batch_size == 1, "Batch size must be 1"

        expert_inputs = {k:v.cuda(self.expert_model.device) for k,v in expert_inputs.items()}
        expert_generated_ids = expert_inputs['input_ids']
        expert_attention_mask = expert_inputs['attention_mask']
        expert_input_len = expert_inputs['input_ids'].shape[1]
        expert_batch_size = expert_inputs['input_ids'].shape[0]
        assert expert_batch_size == 1, "Batch size must be 1"

        num_speculate_tokens = 1
        max_token_len = gen_config.max_new_tokens
        M = max_token_len

        early_stop_flag = False # 用于避免重复校验

        # start_time = time.time()
        decoding_time = 0
        with torch.no_grad():
            # 初始化生成过程
            cur_len=0

            # 这里换用dynamic_M来实现强制早停
            while cur_len < M:
                use_safedecoding = cur_len < self.sd_m

                # Always build the base (victim) model inputs/masks
                raw_output = generated_ids.to(self.model.device)
                prepared_attention_masks = [attention_mask]
                for i in range(num_speculate_tokens):
                    prepared_attention_masks.append(
                        torch.cat(
                            [prepared_attention_masks[i], torch.ones((1, 1), device=attention_mask.device)],
                            dim=-1,
                        )
                    )

                expert_probability_distributions = []

                # Only query expert model for the first m generated tokens
                if use_safedecoding:
                    expert_raw_output = expert_generated_ids.to(self.expert_model.device)
                    expert_prepared_attention_masks = [expert_attention_mask]
                    for i in range(num_speculate_tokens):
                        expert_prepared_attention_masks.append(
                            torch.cat(
                                [
                                    expert_prepared_attention_masks[i],
                                    torch.ones((1, 1), device=expert_attention_mask.device),
                                ],
                                dim=-1,
                            )
                        )

                    # expert(draft) model 预测接下来的 num_speculate_tokens 个 token
                    for _ in range(num_speculate_tokens):
                        outputs = self.expert_model(
                            input_ids=expert_raw_output.to(self.expert_model.device),
                            attention_mask=expert_prepared_attention_masks[_],
                            return_dict=True,
                        )
                        logits = outputs.logits
                        last_token_logits = logits[0, -1, :]
                        probabilities = torch.nn.functional.softmax(last_token_logits, dim=-1)
                        expert_probability_distributions.append(probabilities)

                        # If we ever speculate multiple tokens, extend the draft prompt (except last)
                        next_token = torch.argmax(probabilities, dim=-1)
                        if _ != num_speculate_tokens - 1:
                            expert_raw_output = torch.cat(
                                (expert_raw_output, next_token.unsqueeze(0).unsqueeze(0)), dim=-1
                            )

                # get_speculate_time = time.time()
                # print(f"get_speculate_time: {get_speculate_time - start_time}")

                outputs = self.model(
                    input_ids=raw_output,
                    attention_mask=prepared_attention_masks[-2],
                    return_dict=True
                )
                logits = outputs.logits

                # shape: [bs, seq_len, vocab_size]
                last_token_logits = logits[0, -num_speculate_tokens:, :] #拿出最后的几个需要校验的token对应的logits

                # get_logits_time = time.time()
                # print(f"get_logits_time: {get_logits_time - get_speculate_time}")

                # 转换为概率
                probabilities = torch.nn.functional.softmax(last_token_logits, dim=-1)
                cur_tokens=[]
                break_flag = False
                for i in range(num_speculate_tokens):

                    decoding_start_time = time.time()

                    # Apply SafeDecoding only for the first m generated tokens
                    if use_safedecoding:
                        tmp_idx, _, _, _ = self._select_token(
                            probabilities[i - num_speculate_tokens, :],
                            expert_probability_distributions[i],
                            top_k=self.top_k_merge,
                        )
                        cur_token = tmp_idx
                    else:
                        # Normal decoding after m steps (greedy)
                        cur_token = int(torch.argmax(probabilities[i - num_speculate_tokens, :]).item())

                    decoding_end_time = time.time()
                    decoding_time += decoding_end_time - decoding_start_time
                    # logging for analysis

                    cur_token = torch.Tensor([cur_token]).unsqueeze(0).long().to(generated_ids.device)
                    cur_tokens.append(cur_token)
                    cur_len += 1
                    if(cur_len >= M):
                        break_flag = True
                        break
                    # 终止条件：遇到EOS或达到最大长度
                    if cur_token.item() == self.tokenizer.eos_token_id:
                        break_flag = True
                        break

                # extra_match_tokens +=  len(cur_tokens)-1
                generated_ids = torch.cat([generated_ids]+cur_tokens, dim=-1)


                attention_mask = prepared_attention_masks[len(cur_tokens)]
                expert_generated_ids = torch.cat([expert_generated_ids] + cur_tokens, dim=-1)

                if use_safedecoding:
                    expert_attention_mask = expert_prepared_attention_masks[len(cur_tokens)]
                else:
                    # Keep expert attention mask growing in sync even when expert model is not queried
                    expert_attention_mask = torch.cat(
                        [expert_attention_mask, torch.ones((1, len(cur_tokens)), device=expert_attention_mask.device)],
                        dim=-1,
                    )
                if break_flag:
                    break

        full_texts = [] # the whole conversation texts
        output_texts = [] # the model output part texts
        output_len = 0
        for i, item in enumerate(generated_ids):

            input_pos = inputs['attention_mask'][i].nonzero()

            input_length = input_pos.shape[0] # how many input tokens
            start_pos = input_pos[0][0] # the first non-padding token

            full_text = self.tokenizer.decode(item, skip_special_tokens=True)
            if return_generated_sequence:
                return item[start_pos + input_length:].tolist(),len(item[start_pos + input_length:].tolist())
            output_text = self.tokenizer.decode(item[start_pos + input_length:], skip_special_tokens=True)
            output_len = len(item[start_pos + input_length:])
            # print(f"token per second: {output_len/(end_time - start_time)}")
            full_texts.append(full_text)
            output_texts.append(output_text)
        print(output_texts[0])

        self.reset()    # reset膜电位会比较严谨
        return output_texts[0], output_len

    # 不采用任何defense方法的情况
    def generate_baseline(self, inputs, gen_config=None):
        if gen_config is None:
            gen_config = self.model.generation_config

        if self.verbose:
            logging.info(f"Generation config: {gen_config}")

        inputs = {k:v.cuda(self.model.device) for k,v in inputs.items()}
        # start_time = time.time()
        output_base = self.model.generate(**inputs,
                            generation_config=gen_config,
                            pad_token_id=self.tokenizer.pad_token_id,
                            return_dict_in_generate=True,
                            num_beams=2,
                            length_penalty=1.5,
                            output_scores=True,)
        # end_time = time.time()
        generated_sequence = output_base.sequences[0][inputs["input_ids"].shape[1]:]
        logging.info(f"Generated sequence: {self.tokenizer.decode(generated_sequence)}")
        return self.tokenizer.decode(generated_sequence), len(generated_sequence)
