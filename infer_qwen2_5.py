# import os
# import json
# import torch
# import logging
# import sys
# from datetime import datetime
# from typing import Dict, List, Optional, Union
# from PIL import Image
# from transformers import (
#     AutoModelForImageTextToText,
#     AutoProcessor,
# )
# import moxing as mox

# # ===================== 日志配置 =====================
# def setup_logger(log_dir: str = "./infer_logs"):
#     os.makedirs(log_dir, exist_ok=True)
#     log_path = os.path.join(log_dir, "infer.log")
#     log_format = logging.Formatter(
#         "%(asctime)s - %(levelname)s - %(message)s",
#         datefmt="%Y-%m-%d %H:%M:%S"
#     )
#     root_logger = logging.getLogger()
#     root_logger.handlers = []
#     root_logger.setLevel(logging.INFO)

#     file_handler = logging.FileHandler(log_path, encoding="utf-8")
#     file_handler.setFormatter(log_format)
#     root_logger.addHandler(file_handler)

#     console_handler = logging.StreamHandler(sys.stdout)
#     console_handler.setFormatter(log_format)
#     root_logger.addHandler(console_handler)
#     return logging.getLogger("Qwen-VL-Infer")

# logger = setup_logger()

# # ===================== OBS 图片加载 =====================
# def load_image(image_path: str) -> Image.Image:
#     if image_path.startswith("obs://"):
#         with mox.file.File(image_path, "rb") as f:
#             img = Image.open(f).convert("RGB")
#     else:
#         img = Image.open(image_path).convert("RGB")
#     return img

# # ===================== 推理主函数 =====================
# def infer_qwen_vl(
#     base_model_path: str,
#     checkpoint_path: str,
#     image_paths: Union[str, List[str]],
#     prompt: str = "请详细描述这张图片",
#     max_new_tokens: int = 1024,
#     temperature: float = 0.7,
#     top_p: float = 0.8,
# ):
#     """
#     模型推理函数（支持单图/多图/OBS路径）
#     Args:
#         model_path: 模型路径（训练后的 final_model）
#         image_paths: 图片路径（本地或 obs://）
#         prompt: 问题
#         max_new_tokens: 生成长度
#         temperature: 温度
#         top_p: top_p
#     Returns:
#         response: 模型回答
#     """
#     logger.info("=== 开始推理 ===")
#     logger.info(f"模型路径: {checkpoint_path}")
#     logger.info(f"图片路径: {image_paths}")
#     logger.info(f"用户问题: {prompt}")

#     # 加载处理器 & 模型
#     processor = AutoProcessor.from_pretrained(
#         base_model_path,
#         trust_remote_code=True,
#         padding_side="right"
#     )

#     model = AutoModelForImageTextToText.from_pretrained(
#         checkpoint_path,
#         torch_dtype=torch.bfloat16,
#         device_map="auto",
#         trust_remote_code=True,
#         attn_implementation="flash_attention_2"
#     )
#     model.eval()
#     logger.info("✅ 模型加载完成")

#     # 加载图片
#     if isinstance(image_paths, str):
#         image_paths = [image_paths]
#     images = [load_image(p) for p in image_paths]
#     logger.info(f"✅ 已加载 {len(images)} 张图片")

#     # 构建对话
#     content = []
#     for img in images:
#         content.append({"type": "image", "image": img})
#     content.append({"type": "text", "text": prompt})

#     messages = [
#         {"role": "user", "content": content}
#     ]

#     # 构建输入
#     text = processor.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True
#     )

#     inputs = processor(
#         text=text,
#         images=images if images else None,
#         return_tensors="pt"
#     ).to("npu")

#     # 生成
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             temperature=temperature,
#             top_p=top_p,
#             do_sample=True,
#             pad_token_id=processor.tokenizer.eos_token_id
#         )

#     response = processor.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
#     logger.info(f"模型回答: {response}")
#     return response

# # ===================== 主入口 =====================
# if __name__ == "__main__":
#     # 配置（改成你的路径）
#    # 修改后：
# # 1. 定义两个路径
#     BASE_MODEL_PATH = "/home/ma-user/work/download/models/qwen/Qwen2.5-VL-3B-Instruct"  # 你训练前用的那个原始模型路径
#     CHECKPOINT_PATH = "/home/ma-user/work/outputs/checkpoint-7500"                       # 你训练出来的权重路径
#     IMAGE_PATHS="/home/ma-user/work/siyuanzhang/test_img/3021b0c8a6235b62.jpg"
#     PROMPT = "Suppose you are driving, and I'm providing you with the image captured by the car's front, generate a description of the driving scene which includes the key factors for driving planning, including the positions and movements of vehicles and pedestrians; prevailing weather conditions; time of day, distinguishing between daylight and nighttime; road conditions, indicating smooth surfaces or the presence of obstacles; and the status of traffic lights which influence your decision making, specifying whether they are red or green. The description should be concise, providing an accurate understanding of the driving environment to facilitate informed decision-making."

#     # 推理
#     response = infer_qwen_vl(
#         base_model_path= BASE_MODEL_PATH,
#         checkpoint_path=CHECKPOINT_PATH,
#         image_paths=IMAGE_PATHS,
#         prompt=PROMPT,
#         max_new_tokens=1024,
#         temperature=0.7
#     )

#     # 输出最终结果
#     print("\n" + "="*50)
#     print("最终回答：")
#     print(response)
#     print("="*50)


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
    return logging.getLogger("Qwen-VL-Chat")

logger = setup_logger()

# ===================== OBS 图片加载 =====================
def load_image(image_path: str) -> Image.Image:
    if image_path.startswith("obs://"):
        with mox.file.File(image_path, "rb") as f:
            img = Image.open(f).convert("RGB")
    else:
        img = Image.open(image_path).convert("RGB")
    return img

# ===================== 对话管理器 =====================
class QwenVLChatSession:
    def __init__(self, base_model_path: str, checkpoint_path: str):
        logger.info("正在加载模型和处理器...")
        self.processor = AutoProcessor.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            padding_side="right"
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
        self.model.eval()
        self.messages = []
        self.images = []
        logger.info("✅ 模型加载完成，准备就绪。")

    def chat(self, user_text: str, image_paths: List[str] = None, max_new_tokens: int = 1024):
        """
        进行一轮对话
        """
        # 第一轮对话加载图片
        content = []
        if image_paths and not self.images:
            for p in image_paths:
                img = load_image(p)
                self.images.append(img)
                content.append({"type": "image", "image": img})
            logger.info(f"✅ 已加载 {len(self.images)} 张初始图片")

        # 添加用户文本
        content.append({"type": "text", "text": user_text})
        self.messages.append({"role": "user", "content": content})

        # 构建输入
        text_prompt = self.processor.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Qwen2-VL 处理器需要传入所有历史对话中涉及的图片
        inputs = self.processor(
            text=[text_prompt],
            images=self.images if self.images else None,
            return_tensors="pt"
        ).to(self.model.device) # 自动适配 NPU/GPU

        # 推理生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )

        # 解码
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, outputs)
        ]
        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # 将模型的回答存入历史，以便下一轮对话
        self.messages.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
        
        return response

# ===================== 主入口 =====================
if __name__ == "__main__":
    # 配置路径
    BASE_MODEL_PATH = "/home/ma-user/work/download/models/qwen/Qwen2.5-VL-3B-Instruct"
    CHECKPOINT_PATH = "/home/ma-user/work/outputs/checkpoint-7500"
    # 这里输入你的初始图片路径（可以是单张或列表）
    INIT_IMAGE_PATHS = ["/home/ma-user/work/siyuanzhang/test_img/c01043ae7bfa58f6.jpg"]

    # 初始化会话
    session = QwenVLChatSession(BASE_MODEL_PATH, CHECKPOINT_PATH)

    print("\n" + "*"*30)
    print("系统已进入交互模式。输入 'exit' 或 'quit' 退出，输入 'clear' 重置对话。")
    print("*"*30 + "\n")

    is_first_turn = True

    while True:
        try:
            user_input = input("用户 >> ").strip()
            
            if not user_input:
                continue
            if user_input.lower() in ['exit', 'quit']:
                print("退出对话。")
                break
            if user_input.lower() == 'clear':
                session.messages = []
                session.images = []
                is_first_turn = True
                print("对话历史已清空。下次输入将重新加载图片。")
                continue

            # 第一轮对话会自动带上图片
            if is_first_turn:
                response = session.chat(user_input, image_paths=INIT_IMAGE_PATHS)
                is_first_turn = False
            else:
                # 后续对话仅发送文本
                response = session.chat(user_input)

            print(f"\n模型回答 >> {response}\n")
            print("-" * 50)

        except KeyboardInterrupt:
            print("\n退出对话。")
            break
        except Exception as e:
            logger.error(f"发生错误: {str(e)}", exc_info=True)
            print(f"出错了: {e}")