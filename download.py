import moxing as mox
import PIL.Image
import PIL.ImageOps
import moxing as mox
import base64
import os
from io import BytesIO
import requests

from transformers.utils import (

    requires_backends,
)
# mox.file.copy_parallel('obs://yw-2030-extern//Partner_Sjtu/class_script/sft_infer_qwen2_5','test')

# sub_dirs=mox.file.list_directory('obs://yw-2030-extern/Partner_Zhu/navsim/navsim-data/sensor_blobs/trainval')
# print(len(sub_dirs))
# for d in sub_dirs:
#     print(d)
# import moxing as mox


# obs_path = "obs://yw-2030-extern/Partner_Zhu/navsim/navsim-data/sensor_blobs/trainval/2021.06.09.17.37.09_veh-12_04489_04816/CAM_F0/c01043ae7bfa58f6.jpg"
# local_path = "./siyuanzhang/test_img/c01043ae7bfa58f6.jpg" # 你想保存到本地的路径

# # # 执行拷贝
# mox.file.copy(obs_path, local_path)

# print(f"文件已下载到: {local_path}")
# from modelscope import snapshot_download
# model_dir = snapshot_download(model_id="qwen/Qwen2.5-VL-3B-Instruct",cache_dir="./download/models",revision="master")
import os

# import wandb
# os.environ["WANDB_API_KEY"]="wandb_v1_Q5ZzrTzWfu48zOxZNzmByMXJhcm"
# # 初始化（建议放在训练脚本最上方）
# wandb.init(
#     project="qwen2.5-vl-finetune", # 你的项目名称
#     name="run-v1-recogdrive",      # 实验名称
#     config={
#         "learning_rate": 2e-5,
#         "batch_size": 4,
#         "epochs": 3,
#     }
# )
import moxing as mox

# 1. 路径必须放在引号里，因为它是一个字符串
# 建议末尾加上 / 表示这是一个目录
# target_folder = "obs://yw-2030-extern/Partern_Sjtu/group5/"

# # 2. 调用函数时，括号里放的是变量名，变量名不需要引号
# # 注意：这里要用你上面定义的 target_folder，而不是 group5
# if not mox.file.exists(target_folder):
#     mox.file.make_dirs(target_folder)
#     print(f"✅ 文件夹 {target_folder} 创建成功！")
# else:
#     print(f"ℹ️ 文件夹 {target_folder} 已经存在，无需重复创建。")




#测试obs连接
def load_image(image,timeout):
    """
    Loads `image` to a PIL Image.

    Args:
        image (`str` or `PIL.Image.Image`):
            The image to convert to the PIL Image format.
        timeout (`float`, *optional*):
            The timeout value in seconds for the URL request.

    Returns:
        `PIL.Image.Image`: A PIL Image.
    """
    requires_backends(load_image, ["vision"])
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            # We need to actually check for a real protocol, otherwise it's impossible to use a local file
            # like http_huggingface_co.png
            image = PIL.Image.open(BytesIO(requests.get(image, timeout=timeout).content))
        # patch for obs loading
        elif image.startswith("obs://"):
            with mox.file.File(image, mode='rb') as f:
                img_bytes = f.read()
            image = PIL.Image.open(BytesIO(img_bytes))
        elif os.path.isfile(image):
            image = PIL.Image.open(image)
        else:
            if image.startswith("data:image/"):
                image = image.split(",")[1]

            # Try to load as base64
            try:
                b64 = base64.decodebytes(image.encode())
                image = PIL.Image.open(BytesIO(b64))
            except Exception as e:
                raise ValueError(
                    f"Incorrect image source. Must be a valid URL starting with `http://` or `https://`, a valid path to an image file, or a base64 encoded string. Got {image}. Failed with {e}"
                )
    elif isinstance(image, PIL.Image.Image):
        image = image
    else:
        raise TypeError(
            "Incorrect format used for image. Should be an url linking to an image, a base64 string, a local path, or a PIL image."
        )
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image
test_obs_path = "obs://yw-2030-extern/Partner_Zhu/navsim/navsim-data/sensor_blobs/trainval/2021.06.14.13.27.42_veh-35_04894_05018/CAM_F0/cb2fc291d79f5b66.jpg"
try:
    test_img = load_image(test_obs_path,10.0)
    print(f"✅ OBS图片加载测试成功，图片尺寸: {test_img.size}")
except Exception as e:
    print(f"❌ OBS图片加载测试失败: {e}")