from datasets import load_dataset
import json

dataset = load_dataset("AmazonScience/FalseReject")
test_ds = dataset["test"]

formatted = []

for i, item in enumerate(test_ds):
    formatted.append({
        "id": i,
        "prompt": item["prompt"],
        "category": item["category"],
        "category_text": item["category_text"],
        "instruct_response": item["instruct_response"],
        "cot_response": item["cot_response"]
    })

with open("/home/shensicheng/code/SSD/datasets/FalseReject_test.json", "w", encoding="utf-8") as f:
    json.dump(formatted, f, indent=2, ensure_ascii=False)

print("Saved to test_formatted.json")