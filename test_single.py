import os
import re
import math
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
import moxing as mox

# ===================== 1. 准备单条测试数据 =====================
SAMPLE_DATA = {
    "datasource": "navsim",
    "id": "9310aef3f7e254ea",
    "image": [
        "obs://yw-2030-extern/Partner_Zhu/navsim/navsim-data/sensor_blobs/trainval/2021.05.12.19.36.12_veh-35_00005_00204/CAM_F0/a59241a622955fdc.jpg"
    ],
    "conversations": [
        {
            "from": "human",
            "value": "Here is front views of a driving vehicle:\n<image>\nThe navigation information is: straight\nThe current position is (0.00,0.00)\nCurrent velocity is: (13.48,-0.29)  and current accelerate is: (0.19,0.05)\nPredict the optimal driving action for the next 4 seconds with 8 new waypoints."
        },
        {
            "from": "gpt",
            "value": "6.60,-0.01,13.12,-0.03,19.58,-0.04,25.95,-0.03,32.27,-0.03,38.56,-0.05,44.88,-0.06,51.16,-0.09"
        }
    ]
}

# ===================== 工具函数 =====================
def extract_trajectory_points(text: str) -> list:
    """用正则提取所有数字（包含正负号和浮点数）"""
    numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", text)
    return [float(n) for n in numbers]

def calculate_ade(pred_pts: list, gt_pts: list) -> float:
    """计算 ADE 指标"""
    if len(pred_pts) != 16 or len(gt_pts) != 16:
        print(f"[警告] 轨迹点数量不对！Pred: {len(pred_pts)}, GT: {len(gt_pts)}")
        return 999.0 # 返回一个大数代表失败
    
    total_error = 0.0
    for i in range(8):
        x_pred, y_pred = pred_pts[i*2], pred_pts[i*2 + 1]
        x_gt, y_gt = gt_pts[i*2], gt_pts[i*2 + 1]
        distance = math.sqrt((x_pred - x_gt)**2 + (y_pred - y_gt)**2)
        total_error += distance
    return total_error / 8

# ===================== 主干测试流程 =====================
def test_single_pipeline(base_model_path: str, checkpoint_path: str):
    print("="*50)
    print("🚀 开始单条数据测试流程")
    print("="*50)

    # ---------------- 步骤 A: 解析与准备 ----------------
    print("\n[Step 1/5] 解析数据...")
    obs_image_path = SAMPLE_DATA["image"][0]
    prompt = ""
    gt_text = ""
    for msg in SAMPLE_DATA["conversations"]:
        if msg["from"] == "human":
            prompt = msg["value"].replace("<image>\n", "").replace("<image>", "") 
        elif msg["from"] == "gpt":
            gt_text = msg["value"]
    
    print(f"✅ 图片路径: {obs_image_path}")
    print(f"✅ User Prompt (已剔除<image>): \n{prompt}")
    print(f"✅ Ground Truth (真实标签): {gt_text}")

    # ---------------- 步骤 B: 图片加载与可视化保存 ----------------
    print("\n[Step 2/5] 从 OBS 加载图片并保存到本地...")
    try:
        with mox.file.File(obs_image_path, "rb") as f:
            image = Image.open(f).convert("RGB")
        
        # 保存到当前运行目录，方便你下载查看
        local_save_path = "./test_vis_image.jpg"
        image.save(local_save_path)
        print(f"✅ 图片加载成功！已保存至本地: {local_save_path} (请下载确认图片是否正确)")
    except Exception as e:
        print(f"❌ 图片加载失败: {e}")
        return

    # ---------------- 步骤 C: 模型加载 ----------------
    print("\n[Step 3/5] 正在加载 Qwen2.5-VL 模型与 Processor (这可能需要几分钟)...")
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    model.eval()
    print("✅ 模型加载完成！")

    # ---------------- 步骤 D: 模型推理 ----------------
    print("\n[Step 4/5] 构建输入并进行前向推理...")
    messages = [
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False, # 保证确定性输出
            pad_token_id=processor.tokenizer.eos_token_id
        )
    
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    pred_text = processor.decode(generated_ids, skip_special_tokens=True).strip()
    print(f"✅ 模型预测结果: {pred_text}")

    # ---------------- 步骤 E: 计算 ADE ----------------
    print("\n[Step 5/5] 解析轨迹并计算 ADE...")
    gt_pts = extract_trajectory_points(gt_text)
    pred_pts = extract_trajectory_points(pred_text)
    
    print(f"🔸 解析出的真实坐标 (16个浮点数): {gt_pts}")
    print(f"🔸 解析出的预测坐标 (16个浮点数): {pred_pts}")
    
    if len(gt_pts) == 16 and len(pred_pts) == 16:
        ade_value = calculate_ade(pred_pts, gt_pts)
        print("="*50)
        print(f"🎉 单测完成！计算得到的 ADE = {ade_value:.4f}")
        print("="*50)
    else:
        print("❌ 无法计算 ADE，因为解析出的坐标数量不足 16 个！请检查模型输出格式。")

if __name__ == "__main__":
    # 配置路径 (请替换成你服务器上的实际路径)
    BASE_MODEL_PATH = "/home/ma-user/work/download/models/qwen/Qwen2.5-VL-3B-Instruct"
    CHECKPOINT_PATH = "/home/ma-user/work/outputs/cp7500" 
    
    test_single_pipeline(BASE_MODEL_PATH, CHECKPOINT_PATH)