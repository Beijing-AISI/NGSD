import torch
import torch.nn.functional as F
import logging
import time

class DistributionSafeDecoding:
    def __init__(self, model, tokenizer, expert_model,expert_tokenizer,
                 top_k = 10, num_common_tokens = 10, verbose=False,
                 div_metric: str = "jsd",
                 top_k_merge: int = 64, temperature: float = 1.0,
                 # LIF的超参
                 v_thres: float = 1.0, v_reset: float = 0.0, tau: float = 8.0,
                 # 略大 下降缓慢
                 soft_reset: float = 0.6,
                 # SafeDecoding strength (alpha in logit interpolation)
                 sd_alpha: float = 0.5,
                 **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.expert_model = expert_model
        self.expert_tokenizer = expert_tokenizer
        # self.top_k = top_k
        self.num_common_tokens = num_common_tokens
        self.verbose = verbose
        self.do_sample = False
        self.top_p = None
        # divergence-based parameters
        self.div_metric = div_metric.lower()
        self.top_k_merge = top_k_merge
        self.temperature = temperature
        # LIF neuron state & params
        self.v_thres = float(v_thres)
        self.v_reset = float(v_reset)
        self.tau = float(tau)
        self.v = float(v_reset)
        self.soft_reset = float(soft_reset)
        # SafeDecoding interpolation factor
        self.sd_alpha = float(sd_alpha)
        logging.info("DistributionSafeDecoding with LIF gate initialized.")


    # KL散度

    # 这里需要做Nan值校验！JSD里不需要
    def _kl_div(self, p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        p = torch.nan_to_num(p.float(), nan=0.0, posinf=0.0, neginf=0.0)
        q = torch.nan_to_num(q.float(), nan=0.0, posinf=0.0, neginf=0.0)

        p = p / (p.sum(dim=-1, keepdim=True) + eps)
        q = q / (q.sum(dim=-1, keepdim=True) + eps)

        # 避免 log(0)
        p = torch.clamp(p, min=eps)
        q = torch.clamp(q, min=eps)

        kl = torch.sum(p * (p.log() - q.log()), dim=-1)
        return kl

    # JSD分布 -》 KL散度的平滑版，具有对称性
    def _js_div(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        q = q.to(p.device)
        m = 0.5 * (p + q)
        jsd = 0.5 * self._kl_div(p, m) + 0.5 * self._kl_div(q, m)
        return jsd

    def _divergence(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        if self.div_metric == "kl":
            return self._kl_div(p, q)
        else:
            return self._js_div(p, q)

    def _lif_step(self, v: float, I: float):
        decay = (1.0 - 1.0 / max(self.tau, 1e-6))
        v = decay * v + I
        spiked = v >= self.v_thres
        if spiked:
            # 这里用soft reset
            v = self.v_reset + self.soft_reset * (v - self.v_reset)

        return v, spiked

    def _safe_adjust_topk(self, p_base: torch.Tensor, p_expert: torch.Tensor, top_k: int = None):
        """Apply SafeDecoding-style reweighting on union top-k, using logit interpolation.
        logits' = log_pb + alpha * (log_pe - log_pb) on candidate set; elsewhere keep log_pb.
        Returns best token index and the adjusted score tensor (log-prob like)."""

        p_expert = p_expert.to(p_base.device)
        if top_k is None:
            top_k = self.top_k_merge
        kb = min(top_k, p_base.size(0))
        ke = min(top_k, p_expert.size(0))
        topb = torch.topk(p_base, k=kb).indices.tolist()
        tope = torch.topk(p_expert, k=ke).indices.tolist()
        cand = list(set(topb) | set(tope))

        # 不考虑取对数 对齐SSD
        # log_pb = torch.log(p_base.clamp_min(self.eps))
        # log_pe = torch.log(p_expert.clamp_min(self.eps))
        # start from base
        score = p_base.clone()
        # interpolate in log-space on candidate set
        alpha = self.sd_alpha
        if cand:
            score[cand] = p_base[cand] + alpha * (p_expert[cand] - p_base[cand])
        best_idx = int(torch.argmax(score).item())
        return best_idx, score

    def _select_token(self, p_base: torch.Tensor, p_expert: torch.Tensor, top_k: int = None):
        # # optional temperature smoothing
        # if self.temperature and self.temperature != 1.0:
        #     p_base = torch.softmax(torch.log(p_base.clamp_min(self.eps)) / self.temperature, dim=-1)
        #     p_expert = torch.softmax(torch.log(p_expert.clamp_min(self.eps)) / self.temperature, dim=-1)
        # divergence as input current, normalized to (0,1)

        p_expert = p_expert.to(p_base.device) #保证在一张卡上
        D = self._divergence(p_base, p_expert)
        D_val = float(D.detach().item()) if torch.is_tensor(D) else float(D)

        I = D_val

        # I = D_val / (D_val + 1.0)  # 保证输入电流大小为[0, 1]
        # LIF integrate & fire
        self.v, spiked = self._lif_step(self.v, I)
        if spiked:
            # print("Spiked, using Expert to guide Base model.")
            # print("Current Step Input: I = {}".format(I))
            # print("After Spike: V={}".format(self.v))
            idx, _ = self._safe_adjust_topk(p_base, p_expert, top_k=top_k)
            return idx, True, D_val, float(self.v)
        else:
            idx = int(torch.argmax(p_base).item())
            return idx, False, D_val, float(self.v)

    def reset(self,):
        self.v = float(self.v_reset)
        return

    # 采用SSD
    def safedecoding_speculative(self, inputs, expert_inputs=None, gen_config=None,return_generated_sequence=False):
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

        num_speculate_tokens = 3
        max_token_len = gen_config.max_new_tokens
        M = max_token_len
        g = 7

        # start_time = time.time()
        decoding_time = 0
        with torch.no_grad():
            # 初始化生成过程
            cur_len=0
            # extra_match_tokens=0
            is_token_match_log=[]
            is_safe_decoding_log=[]
            while cur_len < M:
                # start_time = time.time()
                expert_probability_distributions=[]
                expert_speculate_ids = []
                raw_output = generated_ids.to(self.expert_model.device)
                expert_raw_output = expert_generated_ids.to(self.expert_model.device)
                prepared_attention_masks = [attention_mask]
                expert_prepared_attention_masks = [expert_attention_mask]
                for i in range(num_speculate_tokens):
                    prepared_attention_masks.append(torch.cat([prepared_attention_masks[i],torch.ones((1, 1), device=attention_mask.device)],dim=-1))
                    expert_prepared_attention_masks.append(torch.cat([expert_prepared_attention_masks[i],torch.ones((1, 1), device=expert_attention_mask.device)],dim=-1))

                # expert(draft) model 预测接下来的 num_speculate_tokens 个 token
                for _ in range(num_speculate_tokens):
                    # 获取当前 token 的 logits
                    outputs =self.expert_model(
                    input_ids=expert_raw_output.to(self.expert_model.device),
                    attention_mask=expert_prepared_attention_masks[_],
                    return_dict=True
                )
                    logits = outputs.logits

                    # 获取当前最后一个 token 的 logits（假设是 batch_size=1）
                    last_token_logits = logits[0, -1, :]

                    # 将 logits 转换为概率分布
                    probabilities = torch.nn.functional.softmax(last_token_logits, dim=-1)

                    # 保存当前的概率分布 [vocab_size]
                    expert_probability_distributions.append(probabilities)

                    # 贪婪选择最大概率的 token
                    next_token = torch.argmax(probabilities, dim=-1) # 概率最大token的索引

                    expert_speculate_ids.append(next_token.item())

                    # 将选中的 token 添加到生成序列中(不需要最后一个)
                    if _ != num_speculate_tokens-1:
                        raw_output = torch.cat((raw_output, next_token.unsqueeze(0).unsqueeze(0)), dim=-1)
                        expert_raw_output = torch.cat((expert_raw_output, next_token.unsqueeze(0).unsqueeze(0)), dim=-1)
                    # generated_expert_text = self.tokenizer.decode(expert_speculate_ids, skip_special_tokens=True)
                    # print("\nGenerated expert text:", generated_expert_text)

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

                    # 这里开始发生修改
                    decoding_start_time = time.time()
                    tmp_idx, spiked, D_val, v_now = self._select_token(
                        probabilities[i-num_speculate_tokens,:],
                        expert_probability_distributions[i],
                        top_k=self.top_k_merge,
                    )
                    # print("Current V:{}".format(self.v))
                    cur_token = tmp_idx
                    decoding_end_time = time.time()
                    decoding_time += decoding_end_time - decoding_start_time
                    # logging for analysis
                    is_safe_decoding_log.append(1 if spiked else 0)
                    is_token_match_log.append(1 if cur_token == expert_speculate_ids[i] else 0)
                    if self.verbose:
                        mode = "SAFE" if spiked else "BASE"
                        logging.info(f"[{mode}] v={v_now:.3f} D={D_val:.4f} idx={cur_token}")
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
                    if spiked == False:
                        break
                # extra_match_tokens +=  len(cur_tokens)-1
                generated_ids = torch.cat([generated_ids]+cur_tokens, dim=-1)
                attention_mask = prepared_attention_masks[len(cur_tokens)]
                expert_generated_ids = torch.cat([expert_generated_ids]+cur_tokens, dim=-1)
                expert_attention_mask = expert_prepared_attention_masks[len(cur_tokens)]
                # print("extra_match_tokens:",extra_match_tokens)

                # validating_end_time = time.time()
                # print("validating time:",validating_end_time - get_speculate_time)
                # print(f"step {cur_len} total epoch time: {validating_end_time - start_time},extra match tokens:{extra_match_tokens}")
                if break_flag:
                    break
        # generation_end_time = time.time()
        # print(f"generation time: {generation_end_time - function_start_time},time per token:{(generation_end_time - function_start_time)/cur_len}")
            # Split the list into groups of `group_size`
        # end_time = time.time()

        # print("generated_ids:",self.tokenizer.decode(generated_ids[0]))
        # print("expert_generated_ids:",self.tokenizer.decode(expert_generated_ids[0]))
        # raise Exception("stop")
        groups = [is_token_match_log[i:i + g] for i in range(0, len(is_token_match_log), g)]
        ones_ratio = [sum(group) / len(group) for group in groups]

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
        # print(f"Generated sequence: {self.tokenizer.decode(generated_sequence)}")
        # print(f"token per second: {generated_sequence.shape[0]/(end_time - start_time)}")
        return self.tokenizer.decode(generated_sequence), len(generated_sequence)
