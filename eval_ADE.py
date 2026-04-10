import os
import json
import torch
import re
import math
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
import moxing as mox

try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu # 自动将部分 cuda 调用重定向到 npu
except ImportError:
    pass

# ===================== 配置 =====================
BASE_MODEL_PATH = "/home/ma-user/work/download/models/qwen/Qwen2.5-VL-3B-Instruct"
CHECKPOINT_PATH = "/home/ma-user/work/outputs/traj_stage2_model"
EVAL_DATA_PATH = "/home/ma-user/work/outputs/eval_traj.jsonl"  # 你的评测文件路径
NUM_GPUS = 8 # 设置使用的 NPU 卡数

# ===================== 工具函数 =====================
def load_image(image_path: str) -> Image.Image:
    if image_path.startswith("obs://"):
        with mox.file.File(image_path, "rb") as f:
            img = Image.open(f).convert("RGB")
    else:
        img = Image.open(image_path).convert("RGB")
    return img

def extract_points(text: str):
    """
    使用正则表达式从文本中提取 (x, y, h) 坐标点
    返回格式: [(x1, y1), (x2, y2), ...]
    """
    pattern = r"\(([-+]?\d+\.\d+),\s*([-+]?\d+\.\d+),\s*([-+]?\d+\.\d+)\)"
    matches = re.findall(pattern, text)
    points = []
    for m in matches:
        points.append((float(m[0]), float(m[1])))
    return points
 
def calculate_ade(gt_points, pred_points):
    """计算单条数据的 ADE"""
    if len(gt_points) == 0 or len(pred_points) == 0:
        return None
    
    min_len = min(len(gt_points), len(pred_points))
    gt_pts = np.array(gt_points[:min_len])
    pred_pts = np.array(pred_points[:min_len])
    
    distances = np.linalg.norm(gt_pts - pred_pts, axis=1)
    return np.mean(distances)

# ===================== 单卡推理 Worker =====================
def worker_evaluate(rank: int, data_chunk: list, return_dict: dict):
    """
    每个进程执行的函数，在指定的卡（rank）上运行分配到的数据子集
    """
    device = f"npu:{rank}"
    # 可选：显式设置当前设备的 NPU context
    if hasattr(torch, 'npu'):
        torch.npu.set_device(device)

    print(f"[进程 {rank}] 启动成功，分配到 NPU: {rank}，负责 {len(data_chunk)} 条数据。")
    
    # 1. 在当前进程和对应的卡上加载模型
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        CHECKPOINT_PATH,
        torch_dtype=torch.bfloat16,
        device_map=device, # 将模型加载到指定的卡
        trust_remote_code=True
    )
    model.eval()

    results = []
    ade_list = []

    # 2. 遍历推理当前卡负责的数据
    # 使用 position 使得多进程的 tqdm 打印不会互相干扰
    for item in tqdm(data_chunk, desc=f"NPU {rank}", position=rank, leave=False):
        img_path = item["image"]
        gt_text = item["conversations"][2]["value"]
        gt_pts = extract_points(gt_text)
        
        image_obj = load_image(img_path)
        
        sys_p = item["conversations"][0]["value"]
        user_p = item["conversations"][1]["value"].replace("<image>\n", "").replace("<image>", "")
        
        messages = [
            {"role": "system", "content": [{"type": "text", "text": sys_p}]},
            {"role": "user", "content": [
                {"type": "image", "image": image_obj},
                {"type": "text", "text": user_p}
            ]}
        ]

        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = processor(
            text=[text_prompt], images=[image_obj], return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, 
                max_new_tokens=512, 
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
                use_cache=True
            )
            response = processor.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        pred_pts = extract_points(response)
        ade = calculate_ade(gt_pts, pred_pts)
        
        if ade is not None:
            ade_list.append(ade)
            results.append({
                "id": item.get("id"),
                "ade": ade,
                "gt_count": len(gt_pts),
                "pred_count": len(pred_pts)
            })

    # 3. 将当前进程的结果保存到共享字典中
    return_dict[rank] = {
        "ade_list": ade_list,
        "results": results
    }
    print(f"[进程 {rank}] 推理完成！")

# ===================== 主流程与并行调度 =====================
def run_evaluation_multigpu():
    # 注意：在多卡/NPU环境下，必须使用 spawn 启动模式才能正确初始化 CUDA/NPU 上下文
    mp.set_start_method('spawn', force=True)
    
    print(f"正在读取评测数据: {EVAL_DATA_PATH}")
    with open(EVAL_DATA_PATH, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f]
    
    total_samples = len(samples)
    print(f"数据读取完成，共 {total_samples} 条数据。准备启动 {NUM_GPUS} 卡并行推理...")

    # 1. 均分数据集
    chunk_size = math.ceil(total_samples / NUM_GPUS)
    data_chunks = [samples[i:i + chunk_size] for i in range(0, total_samples, chunk_size)]
    actual_gpus = len(data_chunks) # 如果数据量太少，可能分不满 8 份

    # 2. 创建多进程管理器来收集结果
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    # 3. 启动多进程
    for rank in range(actual_gpus):
        p = mp.Process(target=worker_evaluate, args=(rank, data_chunks[rank], return_dict))
        processes.append(p)
        p.start()

    # 4. 等待所有进程结束
    for p in processes:
        p.join()

    # 5. 汇总所有卡的结果
    all_ade_list = []
    all_results = []
    for rank in range(actual_gpus):
        if rank in return_dict:
            all_ade_list.extend(return_dict[rank]["ade_list"])
            all_results.extend(return_dict[rank]["results"])

    # 6. 输出最终统计结果
    if all_ade_list:
        final_ade = np.mean(all_ade_list)
        print("\n" + "="*40)
        print("🎉 8卡并行评测全部完成！")
        print(f"样本总数: {total_samples}")
        print(f"成功计算数: {len(all_ade_list)}")
        print(f"平均 ADE: {final_ade:.4f} 米")
        print("="*40)
        
        with open("eval_results.json", "w", encoding='utf-8') as f:
            json.dump({"average_ade": final_ade, "total_samples": len(all_results), "details": all_results}, f, indent=4)
        print("详细结果已保存至 eval_results.json")
    else:
        print("未成功计算任何 ADE，请检查模型输出或真值格式。")

if __name__ == "__main__":
    run_evaluation_multigpu()