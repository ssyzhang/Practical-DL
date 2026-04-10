import json
import os

def update_image_paths(input_file, output_file):
    # 定义旧路径前缀和新路径前缀
    old_prefix = "obs://yw-2030-extern/Public/Datasets/navsim_dataset/sensor_blobs/"
    new_prefix = "obs://yw-2030-extern/Partner_Zhu/navsim/navsim-data/sensor_blobs/"

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
                    image_val = data["image"]
                    
                    # --- 针对列表格式 ["obs://..."] 的修改逻辑 ---
                    if isinstance(image_val, list) and len(image_val) > 0:
                        # 1. 取出列表中的第一个路径字符串
                        original_path = image_val[0]
                        # 2. 检查并替换前缀
                        if original_path.startswith(old_prefix):
                            new_path = original_path.replace(old_prefix, new_prefix)
                            # 3. 将修改后的字符串放回列表中
                            data["image"] = [new_path]
                            
                    # --- 针对字符串格式 "obs://..." 的兼容逻辑 ---
                    elif isinstance(image_val, str):
                        if image_val.startswith(old_prefix):
                            data["image"] = image_val.replace(old_prefix, new_prefix)
                
                # 将修改后的对象写回文件
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                count += 1
                
            except json.JSONDecodeError as e:
                print(f"跳过错误行: {e}")

    print(f"处理完成！共处理 {count} 条数据。")
    print(f"结果已保存至: {output_file}")

# --- 配置路径 ---
input_jsonl = "/home/ma-user/work/data/navsim_coc_42106_8pts.jsonl"
output_jsonl = "/home/ma-user/work/data/navsim_coc_42106_8pts_renamed.jsonl"

if __name__ == "__main__":
    update_image_paths(input_jsonl, output_jsonl)