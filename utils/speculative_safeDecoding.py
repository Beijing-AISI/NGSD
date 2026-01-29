import torch
import numpy as np
import copy
import logging
import time

class SpeculativeSafeDecoding:
    def _cuda_sync(self, device=None):
        """Synchronize CUDA for accurate timing (no-op on CPU)."""
        if torch.cuda.is_available():
            try:
                if device is None:
                    torch.cuda.synchronize()
                else:
                    torch.cuda.synchronize(device)
            except Exception:
                torch.cuda.synchronize()
    def __init__(self, model, tokenizer, expert_model,expert_tokenizer,  alpha1=0.5,alpha2=0.5,top_k = 10, num_common_tokens = 10, verbose=False):
        self.model = model
        self.tokenizer = tokenizer
        self.expert_model = expert_model
        self.expert_tokenizer = expert_tokenizer
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.top_k = top_k
        self.num_common_tokens = num_common_tokens
        self.verbose = verbose
        self.do_sample = False
        self.top_p = None

        # Aggregated throughput stats (across multiple samples/calls)
        self._agg_samples = 0
        self._agg_total_time = 0.0
        self._agg_total_tokens = 0
        self._agg_t_expert = 0.0
        self._agg_t_base = 0.0
        self._agg_t_select = 0.0
        self._agg_target_samples = 50

        logging.info("SafeDecoding initialized.")
    def naive_spec(self, tensor1, tensor2, C, alpha):
        return 1,torch.topk(tensor1, 1).indices.tolist()[0]
    def safe_decoding(self, tensor1, tensor2, C, alpha):
        tensor2 = tensor2.to(tensor1.device)
        n = min(tensor1.size(0), tensor2.size(0))
        left, right = 1, n
        ans_k = None
        while left <= right:
            mid = (left + right) // 2
            indices1 = set(torch.topk(tensor1, mid).indices.tolist())
            indices2 = set(torch.topk(tensor2, mid).indices.tolist())
            if len(indices1 & indices2) >= C:
                ans_k = mid
                right = mid - 1
            else:
                left = mid + 1

        if ans_k is None:
            return None, None  # 若没有满足条件的 k

        # 获取最终的交集
        topk1 = torch.topk(tensor1, ans_k)
        topk2 = torch.topk(tensor2, ans_k)
        indices1 = set(topk1.indices.tolist())
        indices2 = set(topk2.indices.tolist())
        intersection = indices1 & indices2
        # 在交集内，找到 (tensor1 - tensor2) 差值最大的下标
        max_index = max(intersection, key=lambda idx: tensor1[idx] + alpha*(tensor2[idx] - tensor1[idx]))
        return ans_k, max_index


    # tensor是base和expert的logits
    def safe_decoding_Union(self, tensor1, tensor2, C, alpha):
        tensor2 = tensor2.to(tensor1.device)
        n = min(tensor1.size(0), tensor2.size(0))
        topk1 = torch.topk(tensor1, C)
        topk2 = torch.topk(tensor2, C)
        indices1 = set(topk1.indices.tolist())
        indices2 = set(topk2.indices.tolist())
        intersection = indices1 | indices2
        max_index = max(intersection, key=lambda idx: tensor1[idx] + alpha*(tensor2[idx] - tensor1[idx]))
        return C, max_index

    def safe_decoding_Inter(self, tensor1, tensor2, C, alpha):
        # C是设定的集合大小的阈值

        tensor2 = tensor2.to(tensor1.device)
        n = min(tensor1.size(0), tensor2.size(0))

        # 预先计算所有topk结果，避免重复计算 按照topk排序
        topk1 = torch.topk(tensor1, tensor1.size(0))
        topk2 = torch.topk(tensor2, tensor2.size(0))
        indices1 = topk1.indices
        indices2 = topk2.indices
        # try binary search if C is large

        ans_k = None
        set1 = set()
        set2 = set()


        for k in range(1, n + 1):
            set1.add(indices1[k-1].item())
            set2.add(indices2[k-1].item())
            if len(set1 & set2) >= C:   #逐个加入集合并计算交集
                ans_k = k
                break

        if ans_k is None:
            return None, None

        # 使用预先计算的indices获取交集
        indices1_set = set(indices1[:ans_k].tolist())
        indices2_set = set(indices2[:ans_k].tolist())

        if indices1[0].item() not in indices2_set:
            # 这个分支用于判断高分区域 top1 是否在对方集合
            # 如果高分区域不重合 那么遵循base的输出
            flag = -2
            if indices2[0].item() not in indices1_set:
                flag = 0
            if self.verbose:
                logging.info(f"unmatch:indices2:{indices2[0].item()}: {self.tokenizer.decode(indices2[0].item())},indices1:{indices1[0].item()}: {self.tokenizer.decode(indices1[0].item())},flag:{flag}")
            return flag, indices1[0].item()
        else:
            intersection = indices1_set & indices2_set
            max_index = max(intersection, key=lambda idx: tensor1[idx] + alpha*(tensor2[idx] - tensor1[idx])) # tensor [vocab-size] 对每个词概率的更新
            return 1, max_index


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
        alpha1 = self.alpha1
        alpha2 = self.alpha2
        min_alpha1 = 0.3
        print("alpha1,alpha2:",alpha1,alpha2)
        g = 7
        C = 10
        threshold = 0.6 # change threshold to 0.5 to get better results for Llama3.1
        decoding_mode = 1
        base_threshold = 0.6
        next_decoding_mode = 1

        # Throughput timing (end-to-end) and breakdown
        measure_throughput = bool(kwargs.get("measure_throughput", True))
        timing_breakdown = bool(kwargs.get("timing_breakdown", True))
        warmup_sync = bool(kwargs.get("warmup_sync", True))

        # Aggregate throughput across multiple samples.
        # Default: report every 50 samples.
        aggregate_throughput = bool(kwargs.get("aggregate_throughput", True))
        agg_target_samples = int(kwargs.get("agg_target_samples", 50))
        reset_agg = bool(kwargs.get("reset_agg", False))

        t_total_start = None
        t_expert_sum = 0.0
        t_base_sum = 0.0
        t_select_sum = 0.0

        decoding_time = 0  # legacy: safe_decoding time only

        with torch.no_grad():
            # 初始化生成过程
            cur_len=0
            # extra_match_tokens=0
            is_token_match_log=[]
            is_safe_decoding_log=[]
            if measure_throughput and warmup_sync:
                self._cuda_sync(self.model.device)
                self._cuda_sync(self.expert_model.device)

            if measure_throughput:
                t_total_start = time.perf_counter()

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
                _t0_expert = None
                if measure_throughput and timing_breakdown:
                    self._cuda_sync(self.expert_model.device)
                    _t0_expert = time.perf_counter()
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
                if measure_throughput and timing_breakdown and _t0_expert is not None:
                    self._cuda_sync(self.expert_model.device)
                    t_expert_sum += (time.perf_counter() - _t0_expert)
                # get_speculate_time = time.time()
                # print(f"get_speculate_time: {get_speculate_time - start_time}")

                _t0_base = None
                if measure_throughput and timing_breakdown:
                    self._cuda_sync(self.model.device)
                    _t0_base = time.perf_counter()

                outputs = self.model(
                    input_ids=raw_output,
                    attention_mask=prepared_attention_masks[-2],
                    return_dict=True
                )
                if measure_throughput and timing_breakdown and _t0_base is not None:
                    self._cuda_sync(self.model.device)
                    t_base_sum += (time.perf_counter() - _t0_base)
                logits = outputs.logits

                # shape: [bs, seq_len, vocab_size]
                last_token_logits = logits[0, -num_speculate_tokens:, :] #拿出最后的几个需要校验的token对应的logits

                # get_logits_time = time.time()
                # print(f"get_logits_time: {get_logits_time - get_speculate_time}")

                # 转换为概率
                probabilities = torch.nn.functional.softmax(last_token_logits, dim=-1)
                cur_tokens=[]
                break_flag = False
                _t0_select = None
                if measure_throughput and timing_breakdown:
                    _t0_select = time.perf_counter()
                for i in range(num_speculate_tokens):
                    # 执行算法，贪婪采样
                    correct_flag = False
                    decoding_start_time = time.time()
                    # 第一步默认使用保证utility的交集decoding
                    if decoding_mode == 1:
                        k,cur_token = self.safe_decoding_Inter(tensor1 = probabilities[i-num_speculate_tokens,:],tensor2 =expert_probability_distributions[i],C=C,alpha=alpha1)
                    else:
                        k,cur_token = self.safe_decoding_Union(tensor1 = probabilities[i-num_speculate_tokens,:],tensor2 =expert_probability_distributions[i],C=C,alpha=alpha2)
                    decoding_end_time = time.time()
                    decoding_time += decoding_end_time - decoding_start_time
                    is_safe_decoding_log.append(k)
                    if(cur_token == expert_speculate_ids[i]):
                        correct_flag = True # 可以检查下一步
                        is_token_match_log.append(1) # 这一步match了
                    else:
                        is_token_match_log.append(0)

#---------------------------------------------------------switch decoding_mode,with threshold
                    if len(is_token_match_log) % g == 0:
                        # 每判断7次是否匹配之后 就进行一次decoding模式切换的选择
                        if(sum(is_token_match_log[-g:])/ g < threshold):
                            next_decoding_mode = 2
                        else:
                            next_decoding_mode = 1
                        if next_decoding_mode == decoding_mode:
                            # Fix: Use a small epsilon to handle floating point precision
                            threshold = max(threshold - 0.1, 0)
                            if threshold <= 1e-10:  # If threshold is very close to 0
                                threshold = 0.0
                            if(decoding_mode == 1):
                                alpha1 = max(alpha1-0.15, min_alpha1)
                        else:
                            threshold = base_threshold
                            alpha1 = self.alpha1
                        decoding_mode = next_decoding_mode
                        # print(f"cur_len:{cur_len},alpha:{alpha},threshold:{threshold}")
#---------------------------------------------------------
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
                    if correct_flag == False:
                        break
                if measure_throughput and timing_breakdown and _t0_select is not None:
                    t_select_sum += (time.perf_counter() - _t0_select)
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

        if measure_throughput and t_total_start is not None:
            self._cuda_sync(self.model.device)
            self._cuda_sync(self.expert_model.device)
            t_total = time.perf_counter() - t_total_start
            gen_tokens = int(cur_len)

            # Keep the aggregation target in sync with kwargs (supports changing it at runtime)
            if agg_target_samples > 0:
                self._agg_target_samples = agg_target_samples

            if reset_agg:
                self._agg_samples = 0
                self._agg_total_time = 0.0
                self._agg_total_tokens = 0
                self._agg_t_expert = 0.0
                self._agg_t_base = 0.0
                self._agg_t_select = 0.0

            if aggregate_throughput:
                # accumulate
                self._agg_samples += 1
                self._agg_total_time += float(t_total)
                self._agg_total_tokens += int(gen_tokens)
                if timing_breakdown:
                    self._agg_t_expert += float(t_expert_sum)
                    self._agg_t_base += float(t_base_sum)
                    self._agg_t_select += float(t_select_sum)

                # report every N samples
                if self._agg_samples >= self._agg_target_samples:
                    total_time = max(self._agg_total_time, 1e-9)
                    total_tokens = int(self._agg_total_tokens)
                    tok_per_sec = total_tokens / total_time if total_tokens > 0 else 0.0
                    ms_per_tok = 1000.0 * total_time / max(total_tokens, 1)

                    print("-------------------------------")
                    print(
                        f"[AGG-THROUGHPUT] samples={self._agg_samples}, "
                        f"generated_tokens_total={total_tokens}, total_time={total_time:.4f}s"
                    )
                    print(
                        f"[AGG-THROUGHPUT] avg_tokens_per_sec={tok_per_sec:.2f}, avg_ms_per_token={ms_per_tok:.2f}"
                    )
                    if timing_breakdown:
                        other = total_time - (self._agg_t_expert + self._agg_t_base + self._agg_t_select)
                        print(
                            f"[AGG-BREAKDOWN] expert_forward={self._agg_t_expert:.4f}s, "
                            f"base_forward={self._agg_t_base:.4f}s, select_logic={self._agg_t_select:.4f}s, "
                            f"other={other:.4f}s"
                        )
                    print("-------------------------------")

                    # reset after printing so next batch can be measured too
                    self._agg_samples = 0
                    self._agg_total_time = 0.0
                    self._agg_total_tokens = 0
                    self._agg_t_expert = 0.0
                    self._agg_t_base = 0.0
                    self._agg_t_select = 0.0
            else:
                # Per-sample throughput printing is opt-in
                print_per_sample = bool(kwargs.get("print_per_sample", False))
                if gen_tokens > 0:
                    tok_per_sec = gen_tokens / max(t_total, 1e-9)
                    ms_per_tok = 1000.0 * t_total / max(gen_tokens, 1)
                else:
                    tok_per_sec = 0.0
                    ms_per_tok = 0.0

                if print_per_sample:
                    print("-------------------------------")
                    print(
                        f"[THROUGHPUT] generated_tokens={gen_tokens}, total_time={t_total:.4f}s, "
                        f"tokens/s={tok_per_sec:.2f}, ms/token={ms_per_tok:.2f}"
                    )
                    if timing_breakdown:
                        other = t_total - (t_expert_sum + t_base_sum + t_select_sum)
                        print(
                            f"[BREAKDOWN] expert_forward={t_expert_sum:.4f}s, base_forward={t_base_sum:.4f}s, "
                            f"select_logic={t_select_sum:.4f}s, other={other:.4f}s"
                        )
                    print("-------------------------------")

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

