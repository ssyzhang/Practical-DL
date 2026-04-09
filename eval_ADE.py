# 杀掉所有占用 NPU 的 python 进程
fuser -k /dev/davinci* ```
*(执行完后可以输入 `npu-smi info` 检查一下，确保每张卡的 Memory Usage 都是几十 MB 的干净状态)*

#### 第二步：修改评测代码
我已经针对上述 2 和 3 的问题修改了代码：
1.  **限制图片分辨率**：在读取图片时加入了 `img.thumbnail((1024, 1024))`，防止动态序列过长炸显存。
2.  **移除硬编码的 Flash Attention**：改为使用模型默认的 SDPA，规避升腾底层算子的 Bug。
3.  **强制转换 token_id**：将 `eos_token_id` 转换为安全的 `list` 格式。

请将你的 `eval_batch_multi_npu.py` 或 `test_single.py` 替换为以下代码：

```python
import os
import re
import math
import json
import torch
import logging
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
import moxing as mox
import torch.multiprocessing as mp

try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    pass

# ===================== 日志配置 =====================
def setup_logger(log_dir: str = "./eval_logs", rank: int = 0):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"eval_batch_rank{rank}.log")
    log_format = logging.Formatter(
        f"%(asctime)s - Rank {rank} - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(f"Eval-Batch-{rank}")
    logger.handlers = []
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    
    if rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
        
    return logger

# ===================== 工具函数 =====================
def extract_trajectory_points(text: str) -> list:
    numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", text)
    return [float(n) for n in numbers]

def calculate_ade(pred_pts: list, gt_pts: list) -> float:
    if len(pred_pts) != 16 or len(gt_pts) != 16:
        return -1.0
    
    total_error = 0.0
    for i in range(8):
        x_pred, y_pred = pred_pts[i*2], pred_pts[i*2 + 1]
        x_gt, y_gt = gt_pts[i*2], gt_pts[i*2 + 1]
        distance = math.sqrt((x_pred - x_gt)**2 + (y_pred - y_gt)**2)
        total_error += distance
    return total_error / 8

def load_image_from_obs(image_path: str, logger) -> Image.Image:
    try:
        with mox.file.File(image_path, "rb") as f:
            img = Image.open(f).convert("RGB")
        
        # 【核心修复1】: 限制图片最大分辨率，防止 Qwen2-VL 动态分块导致序列爆炸引发 NPU OOM
        img.thumbnail((1024, 1024)) 
        return img
    except Exception as e:
        logger.error(f"图片加载失败 {image_path}: {e}")
        return None

# ===================== 单个进程的工作函数 =====================
def worker(rank: int, world_size: int, data_chunk: list, base_model_path: str, checkpoint_path: str, tmp_dir: str):
    logger = setup_logger(rank=rank)
    
    device_name = f"npu:{rank}" if hasattr(torch, 'npu') and torch.npu.is_available() else f"cuda:{rank}"
    if 'npu' in device_name:
        torch.npu.set_device(device_name)
    else:
        torch.cuda.set_device(device_name)
        
    logger.info(f"进程启动，分配设备: {device_name}，分发到 {len(data_chunk)} 条数据")

    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    
    # 【核心修复2】: 移除硬编码的 attn_implementation="flash_attention_2"，交由底层自行路由，规避 NPU 长序列 Bug
    if os.path.exists(os.path.join(checkpoint_path, "adapter_config.json")):
        base_model = AutoModelForImageTextToText.from_pretrained(
            base_model_path, torch_dtype=torch.bfloat16, device_map={"": device_name},
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            checkpoint_path, torch_dtype=torch.bfloat16, device_map={"": device_name},
            trust_remote_code=True
        )
    model.eval()

    # 【核心修复3】: 提取并清洗 eos_token_id，防止其为 Tensor 导致生成函数死循环
    eos_ids = processor.tokenizer.eos_token_id
    if isinstance(eos_ids, torch.Tensor):
        eos_ids = eos_ids.tolist()
    elif not isinstance(eos_ids, list):
        eos_ids = [eos_ids]

    valid_count = 0
    total_ade = 0.0
    results_log = []

    pbar = tqdm(data_chunk, desc=f"Rank {rank}", position=rank, unit="条")
    
    for item in pbar:
        data_id = item.get("id", "unknown")
        obs_image_path = item["image"][0]
        
        prompt, gt_text = "", ""
        for msg in item["conversations"]:
            if msg["from"] == "human":
                prompt = msg["value"].replace("<image>\n", "").replace("<image>", "") 
            elif msg["from"] == "gpt":
                gt_text = msg["value"]

        image = load_image_from_obs(obs_image_path, logger)
        if image is None:
            results_log.append({"id": data_id, "error": "Image Load Failed"})
            continue

        try:
            messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
            text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text_input], images=[image], return_tensors="pt").to(device_name)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False, 
                    pad_token_id=processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else eos_ids[0],
                    eos_token_id=eos_ids # 传入清洗后的 eos_ids
                )
            
            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            pred_text = processor.decode(generated_ids, skip_special_tokens=True).strip()

            gt_pts = extract_trajectory_points(gt_text)
            pred_pts = extract_trajectory_points(pred_text)
            
            if len(gt_pts) > 16: gt_pts = gt_pts[:16]
            if len(pred_pts) > 16: pred_pts = pred_pts[:16]

            ade_val = calculate_ade(pred_pts, gt_pts)
            
            if ade_val >= 0:
                total_ade += ade_val
                valid_count += 1
                results_log.append({"id": data_id, "pred": pred_pts, "gt": gt_pts, "ade": round(ade_val, 4)})
            else:
                results_log.append({"id": data_id, "error": "Invalid Output Format", "pred_raw": pred_text})
                
        except Exception as e:
            logger.error(f"推理错误 ID: {data_id} -> {e}")
            results_log.append({"id": data_id, "error": str(e)})

    os.makedirs(tmp_dir, exist_ok=True)
    tmp_file = os.path.join(tmp_dir, f"result_rank_{rank}.json")
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump({"total_ade": total_ade, "valid_count": valid_count, "details": results_log}, f, ensure_ascii=False)
    
    logger.info(f"进程 {rank} 推理结束，已保存临时结果。")

# ===================== 主调度器 =====================
def evaluate_batch_multi_gpu(
    base_model_path: str,
    checkpoint_path: str,
    json_data_path: str,
    output_result_path: str,
    world_size: int = 8,
    max_samples: int = 40
):
    print("="*50)
    print(f"🚀 开始 {world_size} 卡并行批量评测流程")
    print("="*50)

    if json_data_path.startswith("obs://"):
        with mox.file.File(json_data_path, "r") as f:
            data = json.load(f)
    else:
        with open(json_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
    total_samples = len(data)
    print(f"✅ 共读取 {total_samples} 条测试数据，将均分给 {world_size} 张卡...")

    data_chunks = [data[i::world_size] for i in range(world_size)]
    tmp_dir = "./tmp_eval_results"

    mp.set_start_method("spawn", force=True)
    
    processes = []
    for rank in range(world_size):
        if len(data_chunks[rank]) == 0:
            continue 
            
        p = mp.Process(
            target=worker, 
            args=(rank, world_size, data_chunks[rank], base_model_path, checkpoint_path, tmp_dir)
        )
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()

    print("\n\n" + "="*50)
    print("[4/4] 所有进程执行完毕，正在汇总结果...")
    
    global_valid_count = 0
    global_total_ade = 0.0
    global_details = []
    
    for rank in range(world_size):
        tmp_file = os.path.join(tmp_dir, f"result_rank_{rank}.json")
        if os.path.exists(tmp_file):
            with open(tmp_file, "r", encoding="utf-8") as f:
                res = json.load(f)
                global_total_ade += res["total_ade"]
                global_valid_count += res["valid_count"]
                global_details.extend(res["details"])
            os.remove(tmp_file)

    final_mean_ade = (global_total_ade / global_valid_count) if global_valid_count > 0 else -1
    
    summary = {
        "total_samples": total_samples,
        "valid_samples": global_valid_count,
        "failed_samples": total_samples - global_valid_count,
        "mean_ade": final_mean_ade,
        "details": global_details
    }

    with open(output_result_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print(f"🎉 评测汇总报告 🎉")
    print(f"配置: {world_size} NPU 并行")
    print(f"总数据量: {total_samples}")
    print(f"成功计算: {global_valid_count}")
    print(f"失败/异常: {total_samples - global_valid_count}")
    if global_valid_count > 0:
        print(f"🌟 最终平均 ADE 指标: {final_mean_ade:.4f}")
    print(f"详细日志已保存至: {output_result_path}")
    print("="*50)

if __name__ == "__main__":
    BASE_MODEL_PATH = "/home/ma-user/work/download/models/qwen/Qwen2.5-VL-3B-Instruct"
    CHECKPOINT_PATH = "/home/ma-user/work/outputs/cp7500" 
    DATA_JSON_PATH = "/home/ma-user/work/data/navsim_1view_4s_onlytraj_668k.json" 
    OUTPUT_RESULT_PATH = "./batch_eval_ade_multi_npu_results.json"

    evaluate_batch_multi_gpu(
        base_model_path=BASE_MODEL_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        json_data_path=DATA_JSON_PATH,
        output_result_path=OUTPUT_RESULT_PATH,
        world_size=8,
        max_samples=40 
    )