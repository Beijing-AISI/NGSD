from datasets import load_dataset, concatenate_datasets
import pandas as pd

DATASET_NAME = "JailbreakBench/JBB-Behaviors"
OUTPUT_FILE = "../datasets/jbb_all.csv"

def load_and_process(split_name):
    ds = load_dataset(DATASET_NAME, "behaviors",split=split_name)

    def map_label(example):
        if split_name == "benign":
            example["label"] = 1
        else:
            example["label"] = 0
        return example

    return ds.map(map_label)

if __name__ == "__main__":
    train_ds = load_and_process("benign")
    test_ds = load_and_process("harmful")
    all_ds = concatenate_datasets([train_ds, test_ds])

    df = all_ds.to_pandas()

    benign = df[df["label"] == 1].reset_index(drop=True)
    harmful = df[df["label"] == 0].reset_index(drop=True)

    n = min(len(benign), len(harmful))

    alternating_rows = []
    for i in range(n):
        alternating_rows.append(benign.iloc[i])
        alternating_rows.append(harmful.iloc[i])

    # 把多出来的追加在最后
    if len(benign) > n:
        alternating_rows.extend(list(benign.iloc[n:].itertuples(index=False)))
    if len(harmful) > n:
        alternating_rows.extend(list(harmful.iloc[n:].itertuples(index=False)))

    alt_df = pd.DataFrame(alternating_rows)

    # 只保留核心字段（你也可以改）
    keep_cols = [c for c in ["Goal", "label"] if c in alt_df.columns]
    alt_df[keep_cols].to_csv(OUTPUT_FILE, index=False)

    print(f"Saved alternating dataset to: {OUTPUT_FILE}")
    print("Benign:", len(benign), "Harmful:", len(harmful))
    print("Final size:", len(alt_df))