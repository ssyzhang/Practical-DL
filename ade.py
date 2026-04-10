import os
import json
import torch
import re
import numpy as np
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
CHECKPOINT_PATH = "/home/ma-user/work/outputs/recogdrive_stage1_model"
EVAL_DATA_PATH = "/home/ma-user/work/outputs/eval_traj.jsonl"  # 你的评测文件路径
DEVICE = "npu:7" # 8卡环境可以指定 cuda:0 到 7

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
    # 匹配括号内的三个浮点数
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
    
    # 确保点数一致，如果不一致取最小交集（正常模型应输出8个）
    min_len = min(len(gt_points), len(pred_points))
    gt_pts = np.array(gt_points[:min_len])
    pred_pts = np.array(pred_points[:min_len])
    
    # 计算欧几里得距离
    distances = np.linalg.norm(gt_pts - pred_pts, axis=1)
    return np.mean(distances)

# ===================== 推理与评测 =====================
def run_evaluation():
    # 1. 加载模型
    print(f"正在加载模型: {CHECKPOINT_PATH}")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        CHECKPOINT_PATH,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True
    )
    model.eval()

    # 2. 读取数据
    with open(EVAL_DATA_PATH, 'r', encoding='utf-8') as f:
        # 假设是 jsonl 格式，如果是一个大 json list 请用 json.load(f)
        samples = [json.loads(line) for line in f]

    results = []
    ade_list = []

    print(f"开始评测，共 {len(samples)} 条数据...")

    for item in tqdm(samples):
        img_path = item["image"]
        # 获取真值坐标
        gt_text = item["conversations"][2]["value"] # gpt 角色
        gt_pts = extract_points(gt_text)
        
        # 1. 先加载 PIL 图片对象（只加载一次）
        image_obj = load_image(img_path)
        
        # 2. 准备推理 Input
        sys_p = item["conversations"][0]["value"]
        # 清理 human prompt 中的 <image> 标签，processor 会根据 messages 结构自动处理 vision tokens
        user_p = item["conversations"][1]["value"].replace("<image>\n", "").replace("<image>", "")
        
        messages = [
            {"role": "system", "content": [{"type": "text", "text": sys_p}]},
            {"role": "user", "content": [
                {"type": "image", "image": image_obj}, # 这里传入 PIL 对象
                {"type": "text", "text": user_p}
            ]}
        ]

        # 3. 构建 Prompt 文本
        text_prompt = processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        # 4. 执行处理器转换（传入上面加载好的同一个 image_obj）
        inputs = processor(
            text=[text_prompt], 
            images=[image_obj], 
            return_tensors="pt"
        ).to(DEVICE)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False,pad_token_id=processor.tokenizer.eos_token_id,use_cache=True)
            # 截取掉 prompt 长度
            response = processor.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # 解析预测坐标
        pred_pts = extract_points(response)

        # 计算 ADE
        ade = calculate_ade(gt_pts, pred_pts)
        
        if ade is not None:
            ade_list.append(ade)
            results.append({
                "id": item.get("id"),
                "ade": ade,
                "gt_count": len(gt_pts),
                "pred_count": len(pred_pts)
            })
        else:
            print(f"ID {item.get('id')} 解析失败: GT {len(gt_pts)} pts, Pred {len(pred_pts)} pts")

    # 3. 输出统计结果
    if ade_list:
        final_ade = np.mean(ade_list)
        print("\n" + "="*30)
        print(f"评测完成！")
        print(f"样本总数: {len(samples)}")
        print(f"成功计算数: {len(ade_list)}")
        print(f"平均 ADE: {final_ade:.4f} 米")
        print("="*30)
        
        # 可选：保存详细结果
        with open("eval_results.json", "w") as f:
            json.dump({"average_ade": final_ade, "details": results}, f, indent=4)
    else:
        print("未成功计算任何 ADE，请检查正则表达式或模型输出格式。")

if __name__ == "__main__":
    run_evaluation()