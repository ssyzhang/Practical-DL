import json
import os

def split_multi_turn_to_single(input_file, output_file):
    print(f"开始处理: {input_file}")
    
    new_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            item = json.loads(line)
            original_id = item.get("id")
            image = item.get("image")
            convs = item.get("conversations", [])
            
            # 1. 提取系统消息 (如果存在于第一条)
            system_msg = None
            start_idx = 0
            if convs and convs[0]["from"] == "system":
                system_msg = convs[0]
                start_idx = 1
            
            # 2. 遍历对话，两两成对 (human, gpt)
            # 我们从 start_idx 开始，步长为 2 提取 Q&A
            turn_count = 0
            for i in range(start_idx, len(convs), 2):
                # 确保是一对 (human + gpt)
                if i + 1 < len(convs):
                    human_turn = convs[i]
                    gpt_turn = convs[i+1]
                    
                    # 构造新的对话列表
                    new_convs = []
                    if system_msg:
                        new_convs.append(system_msg)
                    
                    # 关键点：单轮对话必须包含 <image> 标签，否则模型不知道看图
                    # 如果这条 human 消息里没带 <image>，我们需要补上
                    if "<image>" not in human_turn["value"]:
                        human_turn["value"] = "<image>\n" + human_turn["value"]
                    
                    new_convs.append(human_turn)
                    new_convs.append(gpt_turn)
                    
                    # 构造单条数据
                    new_item = {
                        "id": f"{original_id}_{turn_count}", # 新 ID
                        "image": image,
                        "conversations": new_convs
                    }
                    new_data.append(new_item)
                    turn_count += 1

    # 写入新文件
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for entry in new_data:
            f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"处理完成！原始数据条数: {len(open(input_file).readlines())}")
    print(f"拆分后总条数: {len(new_data)}")
    print(f"结果保存至: {output_file}")

import json
import random
import os

def merge_and_shuffle_datasets(path_a, path_b, output_path, sample_a_count=8000, seed=42):
    """
    从 A 中随机抽取 sample_a_count 条数据，加入 B 中并彻底打乱
    """
    # 1. 设置随机种子，确保结果可复现
    random.seed(seed)
    
    print(f"正在读取数据集 A: {path_a} ...")
    with open(path_a, 'r', encoding='utf-8') as f:
        lines_a = [line.strip() for line in f if line.strip()]
    
    print(f"正在读取数据集 B: {path_b} ...")
    with open(path_b, 'r', encoding='utf-8') as f:
        lines_b = [line.strip() for line in f if line.strip()]

    # 2. 检查 A 的数据量是否足够
    if len(lines_a) < sample_a_count:
        print(f"警告：数据集 A 只有 {len(lines_a)} 条，不足 {sample_a_count} 条，将使用 A 的全部数据。")
        sampled_a = lines_a
    else:
        # 随机抽取 8000 条
        sampled_a = random.sample(lines_a, sample_a_count)
        print(f"已从 A 中随机抽取 {len(sampled_a)} 条数据。")

    # 3. 合并数据集
    # 此时 combined 包含 100% 的 B 和 抽取的 A
    combined_data = lines_b + sampled_a
    print(f"合并完成。总条数: {len(combined_data)} (B: {len(lines_b)} + A_sample: {len(sampled_a)})")

    # 4. 彻底打乱混合后的数据集
    # 这一步非常重要，确保 A 和 B 的样本在训练过程中交替出现
    random.shuffle(combined_data)
    print("数据集已完成二次打乱（Shuffle）。")

    # 5. 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for line in combined_data:
            f_out.write(line + '\n')
            
    print(f"成功！混合数据集已保存至: {output_path}")



def add_system_prompt_to_jsonl(input_file, output_file, system_value=None):
    """
    为没有 system 提示词的数据集统一添加 system 角色
    """
    # 如果不指定，使用一个通用的自动驾驶感知任务描述
    if system_value is None:
        system_value = "You are a professional autonomous driving perception assistant. Analyze the driving scene images and provide accurate descriptions or reasoning based on the visual information."

    print(f"开始处理: {input_file}")
    
    processed_count = 0
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                convs = data.get("conversations", [])
                
                # 检查第一条是否已经是 system
                if len(convs) > 0 and convs[0]["from"] == "system":
                    # 如果已经有了，可以选择跳过或者替换（这里选择保留原有的）
                    pass
                else:
                    # 构造新的 system 节点
                    new_system = {
                        "from": "system",
                        "value": system_value
                    }
                    # 将 system 插入到对话列表的最前面
                    data["conversations"] = [new_system] + convs
                    processed_count += 1
                
                # 写入新文件
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                
            except Exception as e:
                print(f"处理行报错: {e}")

    print(f"处理完成！")
    print(f"共为 {processed_count} 条数据补充了 system 提示词。")
    print(f"结果保存至: {output_file}")




# --- 配置参数 ---
# 注意：path_a 建议使用你刚刚拆分出来的“单轮对话版”数据集
dataset_a_path = "/home/ma-user/work/data/dataset_navsim_recogdrive_single_system.jsonl" 
dataset_b_path = "/home/ma-user/work/outputs/train_traj.jsonl"
final_output_path = "/home/ma-user/work/outputs/train_traj_mixed.jsonl"

merge_and_shuffle_datasets(
    path_a=dataset_a_path,
    path_b=dataset_b_path,
    output_path=final_output_path,
    sample_a_count=8000,
    seed=42
)





# # --- 运行配置 ---
# # 建议在混合数据集之前，先对 Dataset A 运行这个程序
# input_path = "/home/ma-user/work/data/dataset_navsim_recogdrive_single.jsonl" 
# output_path = "/home/ma-user/work/data/dataset_navsim_recogdrive_single_system.jsonl"

# # 你可以根据 Dataset A 的具体内容自定义这个 system 内容
# my_system_content = "You are a helpful assistant specialized in driving scene understanding and road safety analysis."

# add_system_prompt_to_jsonl(input_path, output_path, system_value=my_system_content)







# # --- 运行 ---
# split_multi_turn_to_single("/home/ma-user/work/data/dataset_navsim_recogdrive_renamed.jsonl", "/home/ma-user/work/data/dataset_navsim_recogdrive_single.jsonl")