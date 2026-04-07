import os
import json
import torch
import logging
import sys
from datetime import datetime
from typing import Dict, List, Optional, Union
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
)
import moxing as mox

# ===================== 日志配置 =====================
def setup_logger(log_dir: str = "./infer_logs"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "infer.log")
    log_format = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(log_format)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    root_logger.addHandler(console_handler)
    return logging.getLogger("Qwen-VL-Infer")

logger = setup_logger()

# ===================== OBS 图片加载 =====================
def load_image(image_path: str) -> Image.Image:
    if image_path.startswith("obs://"):
        with mox.file.File(image_path, "rb") as f:
            img = Image.open(f).convert("RGB")
    else:
        img = Image.open(image_path).convert("RGB")
    return img

# ===================== 推理主函数 =====================
def infer_qwen_vl(
    base_model_path: str,
    checkpoint_path: str,
    image_paths: Union[str, List[str]],
    prompt: str = "请详细描述这张图片",
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.8,
):
    """
    模型推理函数（支持单图/多图/OBS路径）
    Args:
        model_path: 模型路径（训练后的 final_model）
        image_paths: 图片路径（本地或 obs://）
        prompt: 问题
        max_new_tokens: 生成长度
        temperature: 温度
        top_p: top_p
    Returns:
        response: 模型回答
    """
    logger.info("=== 开始推理 ===")
    logger.info(f"模型路径: {checkpoint_path}")
    logger.info(f"图片路径: {image_paths}")
    logger.info(f"用户问题: {prompt}")

    # 加载处理器 & 模型
    processor = AutoProcessor.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        padding_side="right"
    )

    model = AutoModelForImageTextToText.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    model.eval()
    logger.info("✅ 模型加载完成")

    # 加载图片
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    images = [load_image(p) for p in image_paths]
    logger.info(f"✅ 已加载 {len(images)} 张图片")

    # 构建对话
    content = []
    for img in images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": prompt})

    messages = [
        {"role": "user", "content": content}
    ]

    # 构建输入
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=text,
        images=images if images else None,
        return_tensors="pt"
    ).to("npu")

    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=processor.tokenizer.eos_token_id
        )

    response = processor.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    logger.info(f"模型回答: {response}")
    return response

# ===================== 主入口 =====================
if __name__ == "__main__":
    # 配置（改成你的路径）
   # 修改后：
# 1. 定义两个路径
    BASE_MODEL_PATH = "/home/ma-user/work/download/models/qwen/Qwen2.5-VL-3B-Instruct"  # 你训练前用的那个原始模型路径
    CHECKPOINT_PATH = "/home/ma-user/work/outputs/checkpoint-3"                       # 你训练出来的权重路径
    IMAGE_PATHS="/home/ma-user/work/images/00a4c0e63cfa5844.jpg"
    PROMPT = "请描述图中的场景、物体、交通状况"

    # 推理
    response = infer_qwen_vl(
        base_model_path= BASE_MODEL_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        image_paths=IMAGE_PATHS,
        prompt=PROMPT,
        max_new_tokens=1024,
        temperature=0.7
    )

    # 输出最终结果
    print("\n" + "="*50)
    print("最终回答：")
    print(response)
    print("="*50)