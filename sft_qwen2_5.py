


import random
import os
import json
import torch
import base64
import warnings
import requests
import numpy as np
from io import BytesIO
from typing import Dict, List, Optional, Callable, Union
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    # DataCollatorForSeq2Seq,
    # requires_backends
)
from transformers import trainer_utils
from qwen_dataset import QwenLazySupervisedDataset, DataCollatorForSupervisedDataset, make_supervised_data_module,load_image
from datetime import datetime

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
# OBS相关依赖
import moxing as mox

os.environ["WANDB_API_KEY"] = "wandb_v1_Q5ZzrTzWfu48zOxZNzmByMXJhcm"
# 
os.environ["WANDB_PROJECT"] = "qwen2.5-vl-sft-traj"
WANDB_ID = os.environ.get("WANDB_RUN_ID", None)
if WANDB_ID:
    os.environ["WANDB_RESUME"] = "allow"
    os.environ["WANDB_RUN_ID"] = WANDB_ID
    print(f"=== [WandB] 检测到 ID: {WANDB_ID}，将尝试接续曲线 ===")
else:
    os.environ["WANDB_NAME"] = f"run_{current_time}"
    print(f"=== [WandB] 未检测到 ID，将开启新运行: {os.environ['WANDB_NAME']} ===")

# 视频加载相关依赖（按需安装）
try:
    import cv2
    is_cv2_available = True
except ImportError:
    is_cv2_available = False

try:
    import decord
    is_decord_available = True
except ImportError:
    is_decord_available = False

try:
    import av
    is_av_available = True
except ImportError:
    is_av_available = False

try:
    import torchvision
    is_torchvision_available = True
except ImportError:
    is_torchvision_available = False

try:
    import yt_dlp
    is_yt_dlp_available = True
except ImportError:
    is_yt_dlp_available = False

warnings.filterwarnings("ignore")

# ===================== 辅助函数：切分 JSONL =====================
def split_jsonl_file(input_path: str, output_dir: str, eval_ratio: float = 0.05, seed: int = 42):
    """
    按行读取原始 JSONL 文件，打乱后切分为 train 和 eval 两个文件
    """

    train_path = os.path.join(output_dir, "train_split.jsonl")
    eval_path = os.path.join(output_dir, "eval_split.jsonl")
    
    # 【修改】如果文件已存在则不重新切分，保证续训时数据集一致
    if os.path.exists(train_path) and os.path.exists(eval_path):
        print(f"=== 检测到已存在的切分数据集，跳过切分步骤 ===")
        return train_path, eval_path
    print(f"=== 开始划分数据集 (验证集比例: {eval_ratio}) ===")
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 设定随机种子打乱数据，保证每次切分一致
    random.seed(seed)
    random.shuffle(lines)
    
    split_idx = int(len(lines) * (1 - eval_ratio))
    train_lines = lines[:split_idx]
    eval_lines = lines[split_idx:]
    
    # 构建新的路径
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train_split.jsonl")
    eval_path = os.path.join(output_dir, "eval_split.jsonl")
    
    with open(train_path, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    with open(eval_path, 'w', encoding='utf-8') as f:
        f.writelines(eval_lines)
        
    print(f"数据划分完成! 训练集: {len(train_lines)}条, 验证集: {len(eval_lines)}条")
    return train_path, eval_path
def split_jsonl_file_traj(input_path: str, output_dir: str, interval: int = 500):
    """
    按行读取原始 JSONL 文件，每隔 interval 条抽取一条作为 eval，其余作为 train
    """
    train_path = os.path.join(output_dir, "traj_train_split_mixed.jsonl")
    eval_path = os.path.join(output_dir, "traj_eval_split.jsonl")
    
    # 如果文件已存在则不重新切分
    if os.path.exists(train_path) and os.path.exists(eval_path):
        print(f"=== 检测到已存在的切分数据集，跳过切分步骤 ===")
        return train_path, eval_path

    print(f"=== 开始划分数据集 (采样间隔: {interval}) ===")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total_count = len(lines)
    
    # --- 核心修改逻辑 ---
    # 使用切片直接获取评测集 (第0, 500, 1000...条)
    eval_lines = lines[::interval]
    
    # 获取训练集：排除掉那些在 eval_lines 里的索引
    # 这样可以确保数据完全不重叠
    eval_indices = set(range(0, total_count, interval))
    train_lines = [line for i, line in enumerate(lines) if i not in eval_indices]
    # ------------------
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(train_path, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    with open(eval_path, 'w', encoding='utf-8') as f:
        f.writelines(eval_lines)
        
    print(f"数据划分完成! 总数: {total_count} 条")
    print(f"训练集: {len(train_lines)} 条, 验证集: {len(eval_lines)} 条 (采样比例: 1/{interval})")
    
    return train_path, eval_path


# ===================== 6. 训练主函数 =====================
def train_qwen_vl(
    model_path: str,
    train_jsonl_path: str,
    output_dir: str,
    batch_size: int = 2,
    epochs: int = 3,
    lr: float = 5e-5,
):
    # 【新增】1. 划分数据集为 train 和 eval 两个文件
    # 默认留 5% 作为验证集，如果你的数据很少，可把 0.05 调大点（如 0.1）
    train_split_path, eval_split_path = split_jsonl_file_traj(
        input_path=train_jsonl_path, 
        output_dir=output_dir, 
        interval=500  
    )
    last_checkpoint = None
    if os.path.isdir(output_dir):
        last_checkpoint = trainer_utils.get_last_checkpoint(output_dir)
        if last_checkpoint:
            print(f"=== 检测到 Checkpoint: {last_checkpoint}，将从此处恢复训练 ===")
    # 2. 加载模型和处理器
    print("=== 加载模型和处理器 ===")
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right"
    )
    
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        # device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    
    # 显存优化
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # 【修改】3. 分别加载训练集和验证集
    print("=== 构建 Dataset ===")
    train_dataset = QwenLazySupervisedDataset(
        dataset_name='nuscenes_recogdrive',
        processor_path=model_path,
        annotation_path_list=[train_split_path], # 传入切分后的训练文件
        data_root='',
        sampling_rate=1.0,
        git_cfg=dict(patch_size=14, temporal_patch_size=2, merge_size=2)
    )
    
    eval_dataset = QwenLazySupervisedDataset(
        dataset_name='nuscenes_recogdrive',
        processor_path=model_path,
        annotation_path_list=[eval_split_path],  # 传入切分后的验证文件
        data_root='',
        sampling_rate=1.0,
        git_cfg=dict(patch_size=14, temporal_patch_size=2, merge_size=2)
    )

    # 4. 数据整理器
    data_collator = DataCollatorForSupervisedDataset(processor.tokenizer)
    
    # 【修改】5. 训练参数 (加入eval参数)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size, # 添加验证batch_size
        gradient_accumulation_steps=4,
        learning_rate=lr,
        # num_train_epochs=epochs,
        max_grad_norm=0.5,        # <--- 改为 1.0
        warmup_ratio=0.05,
        weight_decay=0.01,
        
        logging_steps=20,         # 每 10 步记录一次 train loss
        save_steps=200,
        eval_strategy="no",    # <--- 新增：按步数进行验证
        eval_steps=200,           # <--- 新增：每 100 步验证一次
        save_total_limit=1,
        max_steps=4000,   # 固定
        num_train_epochs=100,
        fp16=False,
        bf16=True,
        remove_unused_columns=False,
        dataloader_num_workers=2, # <--- 建议改为4，加快数据加载
        dataloader_pin_memory=True,
        lr_scheduler_type="cosine",
        optim="adamw_torch",      # 如果 OOM，考虑改成 "adamw_8bit" 或 "paged_adamw_32bit"
        report_to="wandb",
        seed=42,
        resume_from_checkpoint=last_checkpoint
    )

    # 6. 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # <--- 新增：把验证集传给Trainer
        data_collator=data_collator
    )

    # 7. 开始训练
    print("=== 开始训练 ===")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # 8. 保存模型
    final_model_path = os.path.join(output_dir, "traj_stage2_model")
    trainer.save_model(final_model_path)
    processor.save_pretrained(final_model_path)
    print(f"训练完成！模型保存至: {final_model_path}")

# ===================== 7. 快速启动入口 =====================
if __name__ == "__main__":
    # 配置参数
    MODEL_PATH = "/home/ma-user/work/outputs/recogdrive_stage1_model"
    TRAIN_JSONL_PATH = "/home/ma-user/work/data/dataset_navsim_traj_renamed.jsonl"
    OUTPUT_DIR = "/home/ma-user/work/outputs"

    # OBS连通测试
    test_obs_path = "obs://yw-2030-extern/Partner_Zhu/navsim/navsim-data/sensor_blobs/trainval/2021.06.14.13.27.42_veh-35_04894_05018/CAM_F0/cb2fc291d79f5b66.jpg"
    try:
        test_img = load_image(test_obs_path,10.0)
        print(f"OBS图片加载测试成功，图片尺寸: {test_img.size}")
    except Exception as e:
        print(f"OBS图片加载测试失败: {e}")
    # 启动训练
    train_qwen_vl(
        model_path=MODEL_PATH,
        train_jsonl_path=TRAIN_JSONL_PATH,
        output_dir=OUTPUT_DIR,
        batch_size=1,  
        epochs=3,
        lr=2e-6
    )



