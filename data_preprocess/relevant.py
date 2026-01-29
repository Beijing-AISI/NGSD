import json
import numpy as np
from sentence_transformers import SentenceTransformer

JSON_PATH_1 = "logs/nodefense_vicuna_PAIR.json"   # 比如原始 LLM
JSON_PATH_2 = "logs/NSSD_opt_vicuna_PAIR.json"   # 比如 LLM+A 或 LLM+B

MODEL_NAME= "sentence-transformers/all-MiniLM-L6-v2"

OUTPUT_KEY = "output"   # 每一个id下 代表输出结果的key
ID_KEY     = "id"       # 字段id

NUM_SAMPLES = 50        # 只取前 50 个样本做均值


def load_json_outputs(path, id_key=ID_KEY, output_key=OUTPUT_KEY):
    """
    从 JSON 文件中读取 {id, output} 列表
    返回：ids, texts
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ids = []
    texts = []
    for item in data:
        _id = item.get(id_key)
        out = item.get(output_key, "")

        if out is None:
            out = ""
        ids.append(_id)
        texts.append(str(out))

    return ids, texts


def cosine_sim_batch(x, y):
    """
    x, y: [N, D] 的 numpy 数组
    返回：逐样本余弦相似度 [N]
    """
    x_norm = x / np.linalg.norm(x, axis=1, keepdims=True)
    y_norm = y / np.linalg.norm(y, axis=1, keepdims=True)
    return np.sum(x_norm * y_norm, axis=1)


def main():
    # 1. 读取两个 JSON
    ids1, texts1 = load_json_outputs(JSON_PATH_1)
    ids2, texts2 = load_json_outputs(JSON_PATH_2)

    print(f"文件1 样本数: {len(ids1)}")
    print(f"文件2 样本数: {len(ids2)}")

    # 2. 按 id 对齐（更稳健）：取两边共同的 id，并按 id 排序
    map1 = {i: t for i, t in zip(ids1, texts1)}
    map2 = {i: t for i, t in zip(ids2, texts2)}

    common_ids = sorted(set(map1.keys()) & set(map2.keys()))
    if not common_ids:
        raise ValueError("两个 JSON 没有共同的 id，请检查数据。")

    # 只取前 NUM_SAMPLES 个共同 id
    common_ids = common_ids[:NUM_SAMPLES]
    print(f"共同 id 数量 = {len(common_ids)}（用于计算均值）")

    aligned_texts1 = [map1[i] for i in common_ids]
    aligned_texts2 = [map2[i] for i in common_ids]

    # 3. 加载 SBERT 模型
    print(f"加载模型: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # 4. 编码成向量
    emb1 = model.encode(
        aligned_texts1,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    emb2 = model.encode(
        aligned_texts2,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    # 5. 计算逐样本余弦相似度
    sims = cosine_sim_batch(emb1, emb2)

    # 6. 打印结果
    print("\n===== 结果统计（前 50 个样本）=====")
    print("样本数 N =", len(sims))
    print("相似度均值 =", float(sims.mean()))
    print("相似度标准差 =", float(sims.std()))
    print("相似度最小值 =", float(sims.min()))
    print("相似度最大值 =", float(sims.max()))

    # 如需查看每个 id 的相似度：
    for i, s in zip(common_ids, sims):
        print(f"id = {i}, similarity = {float(s):.4f}")


if __name__ == "__main__":
    main()