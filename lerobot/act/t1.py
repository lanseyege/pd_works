import json
import os, sys

# 方法2：使用生成器，节省内存
def read_jsonl_generator(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():  # 跳过空行
                yield json.loads(line.strip())

# 使用示例
name = "/mnt/nfs_a/cvat-data/zhw/batch3_right/meta/episodes_stats.jsonl"
i = 0
new_file_path = "/home/yuanye/dataset/batch3_right/meta/episodes_stats.jsonl"
with open(new_file_path, 'w', encoding='utf-8') as f:
    for item in read_jsonl_generator(name):
        removed_value = item['stats'].pop("front_top_depth_raw")
        #print(item)
        item_line = json.dumps(item, ensure_ascii=False)
        f.write(item_line + '\n')
        # 处理每个JSON对象
        #print(f"first item over , break, {i}")
        i += 1
        #break
print(f"{i} lines total written!!")
