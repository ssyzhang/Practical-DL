import json
import os

def update_image_paths(input_file, output_file):
    # 定义旧路径前缀和新路径前缀
    # 注意：根据你的数据示例，路径是 ./dataset/sensor_blobs/trainval
    old_prefix = "./dataset/sensor_blobs"
    new_prefix = "obs://yw-2030-extern/Partner_Zhu/navsim/navsim-data/sensor_blobs"

    print(f"开始处理文件: {input_file}")
    
    count = 0
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            
            try:
                # 解析 JSON 对象
                data = json.loads(line)
                
                # 修改 image 字段中的路径
                if "image" in data:
                    original_path = data["image"]
                    # 替换路径前缀
                    if original_path.startswith(old_prefix):
                        data["image"] = original_path.replace(old_prefix, new_prefix)
                
                # 将修改后的对象写回文件
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                count += 1
                
            except json.JSONDecodeError as e:
                print(f"跳过错误行: {e}")

    print(f"处理完成！共处理 {count} 条数据。")
    print(f"结果已保存至: {output_file}")

# --- 配置路径 ---
input_jsonl = "/home/ma-user/work/data/dataset_navsim_traj.jsonl"  # 你的原始文件名
output_jsonl = "/home/ma-user/work/data/dataset_navsim_traj_renamed.jsonl" # 修改后的文件名

update_image_paths(input_jsonl, output_jsonl)