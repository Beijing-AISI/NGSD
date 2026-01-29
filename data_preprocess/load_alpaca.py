from datasets import load_dataset
import json
from pathlib import Path

def save_alpaca_eval_split_to_json(
    dataset_name: str = "tatsu-lab/alpaca_eval",
    split: str = "test",
    output_path: str = "alpaca_eval_test.json",
    indent: int = 2
):
    """
    从 Hugging Face Hub 加载 AlpacaEval 的某个 split，并保存为 JSON 文件。

    参数:
        dataset_name: 数据集名字，例如 "tatsu-lab/alpaca_eval"
        split: 要加载的 split，例如 "train"、"test"，具体要看该数据集提供哪些 split
        output_path: 输出 JSON 文件路径
        indent: JSON 缩进，方便人类阅读
    """
    # 1. 加载数据集和指定 split
    print(f"Loading dataset: {dataset_name}, split: {split}")
    # ds = load_dataset(dataset_name, split=split)
    ds = load_dataset("tatsu-lab/alpaca_eval")

    ds = ds["eval"]
    # 2. 转成 Python list[dict]
    data_list = [dict(row) for row in ds]

    # 3. 确保目录存在
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 4. 保存为 JSON
    print(f"Saving to: {output_path}")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=indent)

    print(f"Done! Saved {len(data_list)} samples to {output_path}")


if __name__ == "__main__":
    # 根据你的需求修改 dataset_name 和 split
    # 比如有的版本可能是 "tatsu-lab/alpaca_eval", split="eval" 或 "test"
    save_alpaca_eval_split_to_json(
        dataset_name="tatsu-lab/alpaca_eval",
        split="eval",                 # 换成你想要的 split 名字
        output_path="/home/shensicheng/code/SSD/datasets/alpaca_eval.json"
    )