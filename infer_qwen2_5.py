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
#     PROMPT = ""

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
#     return logging.getLogger("Qwen-VL-Chat")

# logger = setup_logger()

# # ===================== OBS 图片加载 =====================
# def load_image(image_path: str) -> Image.Image:
#     if image_path.startswith("obs://"):
#         with mox.file.File(image_path, "rb") as f:
#             img = Image.open(f).convert("RGB")
#     else:
#         img = Image.open(image_path).convert("RGB")
#     return img

# # ===================== 对话管理器 =====================
# class QwenVLChatSession:
#     def __init__(self, base_model_path: str, checkpoint_path: str):
#         logger.info("正在加载模型和处理器...")
#         self.processor = AutoProcessor.from_pretrained(
#             base_model_path,
#             trust_remote_code=True,
#             padding_side="right"
#         )
#         self.model = AutoModelForImageTextToText.from_pretrained(
#             checkpoint_path,
#             torch_dtype=torch.bfloat16,
#             device_map="auto",
#             trust_remote_code=True,
#             attn_implementation="flash_attention_2"
#         )
#         self.model.eval()
#         self.messages = []
#         self.images = []
#         logger.info("✅ 模型加载完成，准备就绪。")

#     def chat(self, user_text: str, image_paths: List[str] = None, max_new_tokens: int = 1024):
#         """
#         进行一轮对话
#         """
#         # 第一轮对话加载图片
#         content = []
#         if image_paths and not self.images:
#             for p in image_paths:
#                 img = load_image(p)
#                 self.images.append(img)
#                 content.append({"type": "image", "image": img})
#             logger.info(f"✅ 已加载 {len(self.images)} 张初始图片")

#         # 添加用户文本
#         content.append({"type": "text", "text": user_text})
#         self.messages.append({"role": "user", "content": content})

#         # 构建输入
#         text_prompt = self.processor.apply_chat_template(
#             self.messages,
#             tokenize=False,
#             add_generation_prompt=True
#         )

#         # Qwen2-VL 处理器需要传入所有历史对话中涉及的图片
#         inputs = self.processor(
#             text=[text_prompt],
#             images=self.images if self.images else None,
#             return_tensors="pt"
#         ).to(self.model.device) # 自动适配 NPU/GPU

#         # 推理生成
#         with torch.no_grad():
#             outputs = self.model.generate(
#                 **inputs,
#                 max_new_tokens=max_new_tokens,
#                 do_sample=True,
#                 temperature=0.7,
#                 top_p=0.8,
#                 pad_token_id=self.processor.tokenizer.eos_token_id
#             )

#         # 解码
#         generated_ids = [
#             output_ids[len(input_ids):]
#             for input_ids, output_ids in zip(inputs.input_ids, outputs)
#         ]
#         response = self.processor.batch_decode(
#             generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
#         )[0]

#         # 将模型的回答存入历史，以便下一轮对话
#         self.messages.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
        
#         return response

# # ===================== 主入口 =====================
# if __name__ == "__main__":
#     # 配置路径
#     BASE_MODEL_PATH = "/home/ma-user/work/download/models/qwen/Qwen2.5-VL-3B-Instruct"
#     CHECKPOINT_PATH = "/home/ma-user/work/outputs/traj_stage2_model"
#     # 这里输入你的初始图片路径（可以是单张或列表）
#     INIT_IMAGE_PATHS = ["/home/ma-user/work/siyuanzhang/test_img/c01043ae7bfa58f6.jpg"]

#     # 初始化会话
#     session = QwenVLChatSession(BASE_MODEL_PATH, CHECKPOINT_PATH)

#     print("\n" + "*"*30)
#     print("系统已进入交互模式。输入 'exit' 或 'quit' 退出，输入 'clear' 重置对话。")
#     print("*"*30 + "\n")

#     is_first_turn = True

#     while True:
#         try:
#             user_input = input("用户 >> ").strip()
            
#             if not user_input:
#                 continue
#             if user_input.lower() in ['exit', 'quit']:
#                 print("退出对话。")
#                 break
#             if user_input.lower() == 'clear':
#                 session.messages = []
#                 session.images = []
#                 is_first_turn = True
#                 print("对话历史已清空。下次输入将重新加载图片。")
#                 continue

#             # 第一轮对话会自动带上图片
#             if is_first_turn:
#                 response = session.chat(user_input, image_paths=INIT_IMAGE_PATHS)
#                 is_first_turn = False
#             else:
#                 # 后续对话仅发送文本
#                 response = session.chat(user_input)

#             print(f"\n模型回答 >> {response}\n")
#             print("-" * 50)

#         except KeyboardInterrupt:
#             print("\n退出对话。")
#             break
#         except Exception as e:
#             logger.error(f"发生错误: {str(e)}", exc_info=True)
#             print(f"出错了: {e}")




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

# ===================== OBS 图片加载 (保持不变) =====================
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
    system_prompt: str,  # 新增：系统提示词
    user_prompt: str,    # 用户提示词
    max_new_tokens: int = 1024,
    temperature: float = 0.2, # 轨迹预测建议低温度
    top_p: float = 0.8,
):
    logger.info("=== 开始推理 ===")
    
    # 1. 加载处理器 & 模型
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
    logger.info("✅ 模型及处理器加载完成")

    # 2. 加载图片 (保持你的逻辑)
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    images = [load_image(p) for p in image_paths]
    logger.info(f"✅ 已加载 {len(images)} 张图片")

    # 3. 构建对话结构 (核心修改：加入 System Role)
    # 处理用户 prompt：去掉可能存在的 <image> 字符，因为 processor 处理 images 参数时会自动插入占位符
    clean_user_prompt = user_prompt.replace("<image>\n", "").replace("<image>", "")

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        },
        {
            "role": "user",
            "content": []
        }
    ]
    
    # 将图片和文本放入 user content
    for img in images:
        messages[1]["content"].append({"type": "image", "image": img})
    messages[1]["content"].append({"type": "text", "text": clean_user_prompt})

    # 4. 构建输入
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=images if images else None,
        return_tensors="pt"
    ).to(model.device) # 自动适配 NPU/GPU

    # 5. 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=(temperature > 0),
            pad_token_id=processor.tokenizer.eos_token_id
        )

    # 只解码生成的部分
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    response = processor.decode(generated_ids, skip_special_tokens=True)
    
    logger.info(f"模型回答: {response}")
    return response

# ===================== 主入口 =====================
if __name__ == "__main__":
    # 这里放置你提供的数据样例中的内容
    DATA_SAMPLE = {
        "image": "obs://yw-2030-extern/Partner_Zhu/navsim/navsim-data/sensor_blobs/trainval/2021.07.09.02.42.50_veh-35_00038_02629/CAM_F0/8344b49438605845.jpg",
        "system": "\nYou are a vehicle trajectory prediction model for autonomous driving. Your task is to predict the ego vehicle's 4-second trajectory based on the following inputs: multi-view images from 8 cameras, ego vehicle states (position), and discrete navigation commands. The input provides a 2-second history, and your output should ensure a safe trajectory for the next 4 seconds. Your predictions must adhere to the following metrics:\n1. **No at-fault Collisions (NC)**: Avoid collisions with other objects/vehicles.\n2. **Drivable Area Compliance (DAC)**: Stay within the drivable area.\n3. **Time to Collision (TTC)**: Maintain a safe distance from other vehicles.\n4. **Ego Progress (EP)**: Ensure the ego vehicle moves forward without being stuck.\n5. **Comfort (C)**: Avoid sharp turns and sudden decelerations.\n6. **Driving Direction Compliance (DDC)**: Align with the intended driving direction.\nFor evaluation, use the **PDM Score**, which combines these metrics: **PDM Score** = NC * DAC * (5*TTC + 5*EP + 2*C + 0*DDC) / 12.\nYour predictions will be evaluated through a non-reactive 4-second simulation with an LQR controller and background actors following their recorded trajectories. The better your predictions, the higher your score.\n", # 对应你数据里的 system value
        "human": "<image>\nAs an autonomous driving system, predict the vehicle's trajectory based on:\n1. Visual perception from front camera view\n2. Historical motion context (last 4 timesteps):   - t-3: (-3.03, +0.15, -0.10)    - t-2: (-2.07, +0.08, -0.07)    - t-1: (-1.08, +0.03, -0.04)    - t-0: (0.0, 0.0, 0.0)\n3. Active navigation command: [TURN LEFT]\nOutput requirements:\n- Predict 8 future trajectory points\n- Each point format: (x:float, y:float, heading:float)\n- Use [PT, ...] to encapsulate the trajectory\n- Maintain numerical precision to 2 decimal places" # 对应你数据里的 human value
    }

    BASE_MODEL_PATH = "/home/ma-user/work/download/models/qwen/Qwen2.5-VL-3B-Instruct"
    CHECKPOINT_PATH = "/home/ma-user/work/outputs/traj_stage2_model"

    # 执行推理
    response = infer_qwen_vl(
        base_model_path=BASE_MODEL_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        image_paths=DATA_SAMPLE["image"],
        system_prompt=DATA_SAMPLE["system"],
        user_prompt=DATA_SAMPLE["human"],
        max_new_tokens=512,
        temperature=0.2  # 轨迹预测通常不需要太高随机性
    )

    print("\n" + "="*50)
    print("最终生成的轨迹：")
    print(response)
    print("="*50)