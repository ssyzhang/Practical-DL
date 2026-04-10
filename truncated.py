import json
import os

def process_navsim_jsonl(input_path, output_path):
    processed_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            if not line.strip():
                continue
                
            data = json.loads(line)
            
            # 1. 定位到 GPT 的回答
            for conv in data.get("conversations", []):
                if conv.get("from") == "gpt":
                    original_val = conv.get("value", "")
                    
                    # 2. 拆分 coc 说明和轨迹部分
                    # 格式通常是 "coc:...\ntrajectory:x1,y1,x2,y2..."
                    if "trajectory:" in original_val:
                        parts = original_val.split("trajectory:")
                        prefix = parts[0] + "trajectory:"
                        coords_str = parts[1].strip()
                        
                        # 3. 将坐标按逗号拆分成列表
                        coords_list = coords_str.split(",")
                        
                        # 4. 截取前 8 个点
                        # 一个点包含 x 和 y，所以 8 个点 = 16 个坐标值
                        truncated_coords = coords_list[:16]
                        
                        # 5. 重新拼接
                        new_coords_str = ",".join(truncated_coords)
                        conv["value"] = prefix + new_coords_str
                        
            # 6. 写入新文件
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
            processed_count += 1

    print(f"处理完成！")
    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"总计处理条数: {processed_count}")

# ===================== 执行 =====================
if __name__ == "__main__":
    # 修改为你的文件路径
    INPUT_FILE = "/home/ma-user/work/data/navsim_coc_42106.jsonl"
    OUTPUT_FILE = "/home/ma-user/work/data/navsim_coc_42106_8pts.jsonl"
    
    process_navsim_jsonl(INPUT_FILE, OUTPUT_FILE)