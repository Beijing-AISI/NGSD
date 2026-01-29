# Neuron Guided Safe Decoding

## 🚀 Quick Start

### Environment Setup
**Create virtual environment**
   ```bash
   conda create -n NGSD python=3.10
   conda activate NGSD
   ```
**Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
**Install local packages**
   ```bash
   pip install -e ./peft
   pip install -e ./just_eval
   pip install rouge-score
   ```
**API**

API is needed for training or evaluation, we highly recommend you to export your key in `~/.bashrc`.

### Supported Models
Model Needed:
   - **Vicuna-7B**: `lmsys/vicuna-7b-v1.5`
   - **Llama-2-7B**: `meta-llama/Llama-2-7b-chat-hf`
   - **Llama-2-13B**: `meta-llama/Llama-2-13b-chat-hf`
   - **Qwen3-3-8B**: `Qwen/Qwen3-8B`
   - **TinyLlama**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
   - **DeepAlign Models**: Need to be trained using the [Deep-Align](https://github.com/Unispac/shallow-vs-deep-alignment) repository

### Defense Demo

We **highly recommend** to complete self-reflection via vllm_api but not coupled deeply in decoding code.

Hence, here is a demo to start your NGSD journey by using target model api.
```bash
    CUDA_VISIBLE_DEVICES=0  python -m vllm.entrypoints.openai.api_server \
    --model [your_model] \
    --tensor-parallel-size 1 \
    --max-logprobs 100 \
    --port 8000 \
    --max_model_len 2048 \
    --host 0.0.0.0 \
    --api_key [your_key]
```
Run NGSD as a demo, you should keep your api with the same model and port in the code. You can also choose your prefered Attacker/Defense,
```bash
python defense.py \
  --model_name vicuna \
  --attacker GCG \
  --defender SelfInstinctSafeDecoding \
  --GPT_API [your_api_key] \
  --additional_save_suffix [your_designed_suffix] \
```
