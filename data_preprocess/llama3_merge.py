import json

# 1. 读取原始 JSON 文件
input_file = "../datasets/AutoDAN.json"
output_file = "../datasets/AutDAN.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. 删除 target-model == "llama3" 的项目
filtered_data = [
    item for item in data
    if item.get("target-model") != "llama3"
]

# 3. 写回到新文件（推荐，避免误删原始数据）
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=4)

# 4. 简单统计信息
print(f"原始条目数: {len(data)}")
print(f"删除 llama3 后条目数: {len(filtered_data)}")
print(f"结果已保存至: {output_file}")