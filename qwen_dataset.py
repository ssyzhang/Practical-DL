import json
import random
import logging
import re
import sys
import time
import itertools
from dataclasses import dataclass, field
import typing
import typing_extensions
from typing import Dict, Optional, List, Tuple, Any, Union, Callable
if sys.version_info >= (3, 11):
    Unpack = typing.Unpack
else:
    Unpack = typing_extensions.Unpack
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from mmengine.registry import DATASETS
from mmengine.dataset import BaseDataset

import transformers
transformers_ver = transformers.__version__
from transformers import AutoProcessor, Qwen2_5_VLProcessor

from rope2d import get_rope_index_25, get_rope_index_2, get_rope_index_3

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"
local_rank = None


import base64
import os
from io import BytesIO
import requests
from contextlib import redirect_stdout
from transformers.utils import (
    is_av_available,
    is_cv2_available,
    is_decord_available,
    is_torchvision_available,
    is_yt_dlp_available,
    requires_backends,
)

from transformers.image_utils import is_valid_image
if transformers_ver == '4.50.0':
    from transformers.image_utils import default_sample_indices_fn, VIDEO_DECODERS
else:
    from transformers.video_utils import default_sample_indices_fn, VIDEO_DECODERS
from transformers.processing_utils import (
    TokenizerChatTemplateKwargs,
    ProcessorChatTemplateKwargs,
    AllKwargsForChatTemplate,
    logger,
)

import PIL.Image
import PIL.ImageOps
import moxing as mox


def load_image(image: Union[str, "PIL.Image.Image"], timeout: Optional[float] = None) -> "PIL.Image.Image":
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


def load_video(
            video: Union[str, "VideoInput"],
            num_frames: Optional[int] = None,
            fps: Optional[int] = None,
            backend: str = "opencv",
            sample_indices_fn: Optional[Callable] = None,
            **kwargs,
    ) -> np.array:
        """
        Loads `video` to a numpy array.

        Args:
            video (`str` or `VideoInput`):
                The video to convert to the numpy array format. Can be a link to video or local path.
            num_frames (`int`, *optional*):
                Number of frames to sample uniformly. If not passed, the whole video is loaded.
            fps (`int`, *optional*):
                Number of frames to sample per second. Should be passed only when `num_frames=None`.
                If not specified and `num_frames==None`, all frames are sampled.
            backend (`str`, *optional*, defaults to `"opencv"`):
                The backend to use when loading the video. Can be any of ["decord", "pyav", "opencv", "torchvision"]. Defaults to "opencv".
            sample_indices_fn (`Callable`, *optional*):
                A callable function that will return indices at which the video should be sampled. If the video has to be loaded using
                by a different sampling technique than provided by `num_frames` or `fps` arguments, one should provide their own `sample_indices_fn`.
                If not provided, simple uniformt sampling with fps is performed, otherwise `sample_indices_fn` has priority over other args.
                The function expects at input the all args along with all kwargs passed to `load_video` and should output valid
                indices at which the video should be sampled. For example:

                Example:
                def sample_indices_fn(metadata, **kwargs):
                    return np.linspace(0, metadata.total_num_frames - 1, num_frames, dtype=int)

        Returns:
            Tuple[`np.array`, Dict]: A tuple containing:
                - Numpy array of frames in RGB (shape: [num_frames, height, width, 3]).
                - Metadata dictionary.
        """

        # If `sample_indices_fn` is given, we can accept any args as those might be needed by custom `sample_indices_fn`
        if fps is not None and num_frames is not None and sample_indices_fn is None:
            raise ValueError(
                "`num_frames`, `fps`, and `sample_indices_fn` are mutually exclusive arguments, please use only one!"
            )

        # If user didn't pass a sampling function, create one on the fly with default logic
        if sample_indices_fn is None:

            def sample_indices_fn_func(metadata, **fn_kwargs):
                return default_sample_indices_fn(metadata, num_frames=num_frames, fps=fps, **fn_kwargs)

            sample_indices_fn = sample_indices_fn_func

        if video.startswith("https://www.youtube.com") or video.startswith("http://www.youtube.com"):
            if not is_yt_dlp_available():
                raise ImportError("To load a video from YouTube url you have  to install `yt_dlp` first.")
            # Lazy import from yt_dlp
            requires_backends(load_video, ["yt_dlp"])
            from yt_dlp import YoutubeDL

            buffer = BytesIO()
            with redirect_stdout(buffer), YoutubeDL() as f:
                f.download([video])
            bytes_obj = buffer.getvalue()
            file_obj = BytesIO(bytes_obj)
        elif video.startswith("http://") or video.startswith("https://"):
            file_obj = BytesIO(requests.get(video).content)
        elif video.startswith("obs://"):
            with mox.file.File(video, mode='rb') as f:
                video_bytes = f.read()
            file_obj = BytesIO(video_bytes)
        elif os.path.isfile(video):
            file_obj = video
        elif is_valid_image(video) or (isinstance(video, (list, tuple)) and is_valid_image(video[0])):
            file_obj = None
        else:
            raise TypeError("Incorrect format used for video. Should be an url linking to an video or a local path.")

        # can also load with decord, but not cv2/torchvision
        # both will fail in case of url links
        video_is_url = video.startswith("http://") or video.startswith("https://")
        if video_is_url and backend in ["opencv", "torchvision"]:
            raise ValueError(
                "If you are trying to load a video from URL, you can decode the video only with `pyav` or `decord` as backend"
            )

        if file_obj is None:
            return video

        if (
                (not is_decord_available() and backend == "decord")
                or (not is_av_available() and backend == "pyav")
                or (not is_cv2_available() and backend == "opencv")
                or (not is_torchvision_available() and backend == "torchvision")
        ):
            raise ImportError(
                f"You chose backend={backend} for loading the video but the required library is not found in your environment "
                f"Make sure to install {backend} before loading the video."
            )

        video_decoder = VIDEO_DECODERS[backend]
        video, metadata = video_decoder(file_obj, sample_indices_fn, **kwargs)
        return video, metadata


def sample_indices_fn_with_limit(metadata, fps=2.0, max_frames=20, **kwargs):
    """
    Video sampling function: samples frames uniformly based on the fps, but limits the maximum number of frames.

    Args:
        metadata: including total_num_frames and fps.
        fps: Target sampling frame rate (default: 2.0).
        max_frames: Maximum number of sampled frames (default: 20).

    Returns:
        np.ndarray: Indexes of the sampled frames.
    """
    total_frames = metadata.total_num_frames
    video_fps = getattr(metadata, 'fps', 2.0)  # default original video frame rate

    if video_fps <= 0:
        video_fps = 30.0

    video_duration = total_frames / video_fps

    # number of frames required for calculating the FPS
    desired_frames = int(video_duration * fps) if video_duration > 0 else 1

    # Return the frame ids with the specified FPS
    if desired_frames < max_frames:
        return np.linspace(0, total_frames - 1, desired_frames, dtype=int)

    sample_count = min(max(desired_frames, 1), max_frames)

    metadata.fps = sample_count / video_duration

    if total_frames <= 1:
        return np.array([0])

    return np.linspace(0, total_frames - 1, sample_count, dtype=int)


def apply_chat_template(
        self,
        conversation: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
        chat_template: Optional[str] = None,
        **kwargs: Unpack[AllKwargsForChatTemplate],
    ) -> str:
        """
        Similar to the `apply_chat_template` method on tokenizers, this method applies a Jinja template to input
        conversations to turn them into a single tokenizable string.

        The input is expected to be in the following format, where each message content is a list consisting of text and
        optionally image or video inputs. One can also provide an image, video, URL or local path which will be used to form
        `pixel_values` when `return_dict=True`. If not provided, one will get only the formatted text, optionally tokenized text.

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "https://www.ilankelman.org/stopsigns/australia.jpg"},
                    {"type": "text", "text": "Please describe this image in detail."},
                ],
            },
        ]

        Args:
            conversation (`Union[List[Dict, [str, str]], List[List[Dict[str, str]]]]`):
                The conversation to format.
            chat_template (`Optional[str]`, *optional*):
                The Jinja template to use for formatting the conversation. If not provided, the tokenizer's
                chat template is used.
        """
        if chat_template is None:
            if self.chat_template is not None:
                chat_template = self.chat_template
            else:
                raise ValueError(
                    "No chat template is set for this processor. Please either set the `chat_template` attribute, "
                    "or provide a chat template as an argument. See "
                    "https://huggingface.co/docs/transformers/main/en/chat_templating for more information."
                )

        # Fill two sets of kwargs that should be used by tokenizer's `apply_chat_template`
        # and for multimodal chat template
        tokenizer_template_kwargs = {}
        for tokenizer_key in TokenizerChatTemplateKwargs.__annotations__.keys():
            tokenizer_value = getattr(TokenizerChatTemplateKwargs, tokenizer_key, None)
            value = kwargs.pop(tokenizer_key, tokenizer_value)
            tokenizer_template_kwargs[tokenizer_key] = value

        chat_template_kwargs = {}
        for key in ProcessorChatTemplateKwargs.__annotations__.keys():
            processor_value = getattr(ProcessorChatTemplateKwargs, key, None)
            value = kwargs.pop(key, processor_value)
            chat_template_kwargs[key] = value

        if isinstance(conversation, (list, tuple)) and (
            isinstance(conversation[0], (list, tuple)) or hasattr(conversation[0], "content")
        ):
            is_batched = True
            conversations = conversation
        else:
            is_batched = False
            conversations = [conversation]

        num_frames = chat_template_kwargs.get("num_frames")
        video_fps = chat_template_kwargs.get("video_fps", 2)
        video_load_backend = chat_template_kwargs.get("video_load_backend", 'pyav')
        video_max_frames = chat_template_kwargs.get("video_max_frames", 20)
        tokenize = chat_template_kwargs.get("tokenize")
        return_dict = chat_template_kwargs.get("return_dict")
        sample_indices_fn = chat_template_kwargs.get("sample_indices_fn")

        # default function with a maximum frame limit is used.
        if sample_indices_fn is None:
            sample_indices_fn = sample_indices_fn_with_limit

        if tokenize:
            batch_images, batch_videos = [], []
            batch_video_metadata = []
            for conversation in conversations:
                images, videos = [], []
                video_metadata = []
                for message in conversation:
                    visuals = [content for content in message["content"] if content["type"] in ["image", "video"]]
                    image_fnames = [
                        vision_info[key]
                        for vision_info in visuals
                        for key in ["image", "url", "path", "base64"]
                        if key in vision_info and vision_info["type"] == "image"
                    ]
                    video_fnames = [
                        vision_info[key]
                        for vision_info in visuals
                        for key in ["video", "url", "path"]
                        if key in vision_info and vision_info["type"] == "video"
                    ]
                    for fname in image_fnames:
                        images.append(load_image(fname))
                    for fname in video_fnames:
                        if isinstance(fname, (list, tuple)) and isinstance(fname[0], str):
                            video = [np.array(load_image(image_fname)).T for image_fname in fname]
                            # create a 4D video because `load_video` always returns a 4D array
                            video = np.stack(video)
                            metadata = None
                            logger.warning(
                                "When loading the video from list of images, we cannot infer metadata such as `fps` or `duration`. "
                                "If you model applies special processing based on metadata, please load the whole video and let the model sample frames."
                            )
                        else:
                            video, metadata = load_video(
                                fname,
                                num_frames=num_frames,
                                fps=video_fps,
                                backend=video_load_backend,
                                sample_indices_fn=sample_indices_fn,
                                max_frames=video_max_frames,
                            )
                        videos.append(video)
                        video_metadata.append(metadata)

                # Currently all processors can accept nested list of batches, but not flat list of visuals
                # So we'll make a batched list of images and let the processor handle it
                if images:
                    batch_images.append(images)
                if videos:
                    batch_videos.append(videos)
                    batch_video_metadata.append(video_metadata)

            # Process conversation with video/image information if needed. Then convert into a prompt using Jinja template
            # conversations = self._process_messages_for_chat_template(
            #     conversations,
            #     batch_images=batch_images,
            #     batch_videos=batch_videos,
            #     batch_video_metadata=batch_video_metadata,
            #     **chat_template_kwargs,
            # )

        prompt = self.tokenizer.apply_chat_template(
            conversations,
            chat_template=chat_template,
            tokenize=False,
            return_dict=False,
            **tokenizer_template_kwargs,
        )

        if not is_batched:
            prompt = prompt[0]

        if tokenize:
            # Tokenizer's `apply_chat_template` never adds special tokens when tokenizing
            # But processor's `apply_chat_template` didn't have an option to tokenize, so users had to format the prompt
            # and pass it to the processor. Users thus never worried about special tokens relying on processor handling
            # everything internally. The below line is to keep BC for that and be able to work with model that have
            # special tokens in the template (consistent with tokenizers). We dont want to raise warning, it will flood command line
            # without actionable solution for users
            single_prompt = prompt[0] if is_batched else prompt
            if self.tokenizer.bos_token is not None and single_prompt.startswith(self.tokenizer.bos_token):
                kwargs["add_special_tokens"] = False

            out = self(
                text=prompt,
                images=batch_images if batch_images else None,
                videos=batch_videos if batch_videos else None,
                **kwargs,
            )
            if return_dict:
                out['metadata'] = batch_video_metadata
                return out
            else:
                return out["input_ids"]
        return prompt


setattr(Qwen2_5_VLProcessor, 'apply_chat_template', apply_chat_template)


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def _make_abs_paths(base: Path, files: str) -> str:
    return os.path.join(base, files)


def update_processor_pixels(processor,
                            min_pixels, max_pixels,
                            video_max_pixels, video_min_pixels,
                            video_max_frames, video_min_frames,
                            video_fps):
    logger = logging.getLogger(__name__)

    # --- Image Processor ---
    ip = processor.image_processor
    rank0_print("=== BEFORE IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"ip.size: {ip.size}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    if hasattr(ip, "min_pixels") and hasattr(ip, "max_pixels"):
        ip.min_pixels = min_pixels
        ip.max_pixels = max_pixels
        rank0_print(f"✅ Updated image_processor min_pixels to {min_pixels}")
        rank0_print(f"✅ Updated image_processor max_pixels to {max_pixels}")

    if hasattr(ip, "size") and isinstance(ip.size, dict):
        ip.size["shortest_edge"] = min_pixels
        ip.size["longest_edge"] = max_pixels
        rank0_print(
            f"✅ Updated image_processor size['shortest_edge'] to {min_pixels}"
        )
        rank0_print(
            f"✅ Updated image_processor size['longest_edge'] to {max_pixels}"
        )

    rank0_print("=== AFTER IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    # --- Video Processor ---
    if hasattr(processor, "video_processor") and processor.video_processor is not None:
        vp = processor.video_processor
        rank0_print("\n=== BEFORE VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        rank0_print(
            f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
        )
        rank0_print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

        if hasattr(vp, "min_pixels") and hasattr(vp, "max_pixels"):
            vp.min_pixels = video_min_pixels
            vp.max_pixels = video_max_pixels
            rank0_print(
                f"✅ Updated Qwen2-VL video_processor min_pixels to {video_min_pixels}"
            )
            rank0_print(
                f"✅ Updated Qwen2-VL video_processor max_pixels to {video_max_pixels}"
            )

        if hasattr(vp, "min_frames") and hasattr(vp, "max_frames"):
            vp.min_frames = video_min_frames
            vp.max_frames = video_max_frames
            rank0_print(
                f"✅ Updated video_processor min_frames to {video_min_frames}"
            )
            rank0_print(
                f"✅ Updated video_processor max_frames to {video_max_frames}"
            )

        if hasattr(vp, "fps"):
            vp.fps = video_fps
            rank0_print(f"✅ Updated video_processor fps to {video_fps}")

        if hasattr(vp, "size") and isinstance(vp.size, dict):
            vp.size["shortest_edge"] = video_min_pixels
            vp.size["longest_edge"] = video_max_pixels
            rank0_print(
                f"✅ Updated Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
            )
            rank0_print(
                f"✅ Updated Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}"
            )

        rank0_print("=== AFTER VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        rank0_print(
            f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
        )
        rank0_print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

    return processor


def _estimate_token_count(text: str, multiplier: float = 0.7) -> int:
    """
    Estimates the number of tokens in a text.
    In the mixed Chinese and English scenario, the average number of tokens per character is about 0.75 for English and 0.5 for Chinese.
    Here, the average multiplier 0.7 is used as an empirical value.

    Args:
        text: Input text
        multiplier: Token estimation multiplier

    Returns:
        Estimated number of tokens
    """
    return int(len(text) * multiplier)


def _smart_truncate_conversations(
    item: Dict[str, Any],
    max_turns: Optional[int] = None,
    max_tokens: Optional[int] = None,
    random_sampling: bool = False,
) -> Dict[str, Any]:
    """
    Truncate conversations, prioritizing the first N rounds of dialogue.

    Strategy:
    1. Estimate the total number of tokens.If it does not exceed max_tokens, retain all tokens.
    2. If it exceeds, check if the number of dialogue turns exceeds max_turns.
    3. If it exceeds, retain the first max_turns rounds of dialogue.
    4. If random_sampling=True, randomly sample from the multi-round dialogue.

    Args:
        max_turns: Maximum number of dialogue turns, None means no limit.
        max_tokens: Maximum number of tokens, None means no limit.
        random_sampling: Whether to sample randomly (only takes effect when the number of turns exceeds the limit).

    Returns:
        Processed data sample (a copy is returned, the original data is not modified).
    """
    conversations = item["conversations"]

    if max_turns is None and max_tokens is None:
        return item

    total_turns = len(conversations)

    # Estimate the total number of tokens
    total_tokens = sum(_estimate_token_count(turn["value"]) for turn in conversations)

    # Token estimation for adding images or videos.
    if "image" in item:
        images = item["image"] if isinstance(item["image"], list) else [item["image"]]
        total_tokens += len(images) * 256  # about 256 tokens each image (estimated based on the patch size).
    if "video" in item:
        videos = item["video"] if isinstance(item["video"], list) else [item["video"]]
        total_tokens += len(videos) * 512  # about 512 tokens each video

    # If the limit is not exceeded, return directly.
    if max_tokens is not None and total_tokens <= max_tokens:
        return item

    if max_turns is not None and total_turns <= max_turns:
        return item

    # Truncate conversations
    if max_turns is not None and total_turns > max_turns:
        if random_sampling:
            # random_sampling
            sampled_indices = sorted(random.sample(range(total_turns), max_turns))
            truncated_conversations = [conversations[i] for i in sampled_indices]
        else:
            # first N rounds of dialogue (considering system, human, gpt, ...)
            if conversations[0]["from"] == "system":
                max_turns += 1
            truncated_conversations = conversations[:max_turns]

        result = item.copy()
        result["conversations"] = truncated_conversations

        # the number of images/videos should be adjusted accordingly
        if "image" in item or "video" in item:
            _adjust_media_for_truncated_conversations(result)

        return result

    return item


def _adjust_media_for_truncated_conversations(item: Dict[str, Any]):
    """
    Adjust the number of images or videos based on the truncated conversation to ensure the number of placeholders matches the number of actual media files.
    Only keep the media files actually used in the conversation, and match the media files from front to back.
    """
    conversations = item["conversations"]

    # Number of images or videos required in the conversation after truncation.
    image_placeholders = 0
    video_placeholders = 0

    for turn in conversations:
        text = turn["value"]
        image_placeholders += text.count("<image>")
        video_placeholders += text.count("<video>")

    # Adjusting images
    if "image" in item:
        images = item["image"] if isinstance(item["image"], list) else [item["image"]]
        if len(images) > image_placeholders:
            item["image"] = images[:image_placeholders] if image_placeholders > 0 else []

    # Adjusting videos
    if "video" in item:
        videos = item["video"] if isinstance(item["video"], list) else [item["video"]]
        if len(videos) > video_placeholders:
            item["video"] = videos[:video_placeholders] if video_placeholders > 0 else []


def _build_messages(item: Dict[str, Any], base_path: Path) -> List[Dict[str, Any]]:
    # Extract and normalize images and videos
    images = item.get("image") or []
    if isinstance(images, str):
        images = [images]

    videos = item.get("video") or []
    if isinstance(videos, str):
        videos = [videos]

    # Build media pools with absolute paths
    image_pool = [
        {"type": "image", "image": _make_abs_paths(base_path, img)} for img in images
    ]
    video_pool = [
        {"type": "video", "video": _make_abs_paths(base_path, vid)} for vid in videos
    ]

    messages = []
    for turn in item["conversations"]:
        if turn["from"] == "human":
            role = "user"
        elif turn["from"] == "gpt":
            role = "assistant"
        elif turn["from"] == "system":
            role = "system"
        else:
            raise ValueError(f"Unknown role: {turn['from']}")
        text: str = turn["value"]

        if role == "user":
            content = []
            # Split text by <image> or <video> placeholders while keeping delimiters
            text_parts = re.split(r"(<image>|<video>)", text)

            for seg in text_parts:
                if seg == "<image>":
                    if not image_pool:
                        raise ValueError(
                            "Number of <image> placeholders exceeds the number of provided images"
                        )
                    content.append(image_pool.pop(0))
                elif seg == "<video>":
                    if not video_pool:
                        raise ValueError(
                            "Number of <video> placeholders exceeds the number of provided videos"
                        )
                    content.append(video_pool.pop(0))
                elif seg.strip():
                    content.append({"type": "text", "text": seg.strip()})

            messages.append({"role": role, "content": content})
        else:
            # Assistant messages contain only text
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})

    # Check for unused media files
    if image_pool:
        raise ValueError(
            f"{len(image_pool)} image(s) remain unused (not consumed by placeholders)"
        )
    if video_pool:
        raise ValueError(
            f"{len(video_pool)} video(s) remain unused (not consumed by placeholders)"
        )

    return messages


def preprocess_qwen_visual(
    sources,
    processor,
    video_max_frames: int = 20,
) -> Dict:
    if len(sources) != 1:
        raise ValueError(f"Expected 1 source, got {len(sources)}")

    source = sources[0]
    base_path = source.get("data_path", "")
    if 'image' not in source and 'video' not in source:
        raise ValueError('No image or video in source')
    messages = _build_messages(source, base_path)

    full_result = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt",
        video_fps=2, video_max_frames=video_max_frames
    )

    input_ids = full_result["input_ids"]
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids).unsqueeze(0)

    labels = torch.full_like(input_ids, IGNORE_INDEX)

    input_ids_flat = input_ids[0].tolist()
    L = len(input_ids_flat)
    pos = 0
    while pos < L:
        if input_ids_flat[pos] == 77091:
            ans_start = pos + 2
            ans_end = ans_start
            while ans_end < L and input_ids_flat[ans_end] != 151645:
                ans_end += 1
            if ans_end < L:
                labels[0, ans_start : ans_end + 2] = input_ids[
                    0, ans_start : ans_end + 2
                ]
                pos = ans_end
        pos += 1

    full_result["labels"] = labels
    full_result["input_ids"] = input_ids
    return full_result


@DATASETS.register_module()
class QwenLazySupervisedDataset(BaseDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self,
                 dataset_name,
                 processor_path,
                 annotation_path_list: [str] = None,
                 data_root: str = None,
                 sampling_rate: float = None,
                 model_type: str = "qwen2.5vl",
                 max_pixels: int = 28 * 28 * 576,
                 min_pixels: int = 28 * 28 * 16,
                 video_max_frames: int = 20,
                 video_min_frames: int = 4,
                 video_max_pixels: int = 576 * 28 * 28,
                 video_min_pixels: int = 16 * 28 * 28,
                 video_max_total_pixels: int = None,
                 video_min_total_pixels: int = None,
                 video_fps: float = 2,
                 image_max_total_pixels: int = 10 * 672 * 672,
                 data_packing: bool = False,
                 git_cfg: dict = None,
                 # Truncation Parameters
                 max_turns: int = 50,
                 max_tokens: int = 6000,
                 random_turn_sampling: bool = False,
                 ):
        super(QwenLazySupervisedDataset, self).__init__(serialize_data=False)

        # dataset = data_args.dataset_use.split(",")
        # dataset_list = data_list(dataset)
        # rank0_print(f"Loading datasets: {dataset_list}")
        self.git_cfg = git_cfg
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.video_max_frames = video_max_frames
        self.video_max_total_pixels = video_max_total_pixels if video_max_total_pixels is not None else 1664 * 28 * 28
        self.video_min_total_pixels = video_min_total_pixels if video_min_total_pixels is not None else 256 * 28 * 28
        self.image_max_total_pixels = image_max_total_pixels if image_max_total_pixels is not None else 10 * 672 * 672
        self.model_type = model_type
        # Truncation Parameters
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.random_turn_sampling = random_turn_sampling
        if model_type == "qwen3vl":
            self.get_rope_index = get_rope_index_3
        elif model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        elif model_type == "qwen2vl":
            self.get_rope_index = get_rope_index_2
        else:
            raise ValueError(f"model_type: {model_type} not supported")

        list_data_dict = []

        for annotation_path in annotation_path_list:
            file_format = annotation_path.split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(annotation_path)
            else:
                annotations = json.load(open(annotation_path, "r"))

            if sampling_rate < 1.0:
                annotations = random.sample(
                    annotations, int(len(annotations) * sampling_rate)
                )
                rank0_print(f"sampling {len(annotations)} examples from dataset {dataset_name}")
            else:
                rank0_print(f"dataset name: {dataset_name}")
            for ann in annotations:
                if isinstance(ann, list):
                    for sub_ann in ann:
                        sub_ann["data_path"] = data_root
                        sub_ann["dataset_name"] = dataset_name
                else:
                    ann["data_path"] = data_root
                    ann["dataset_name"] = dataset_name
            list_data_dict += annotations

        rank0_print(f"Total training samples: {len(list_data_dict)}")

        random.shuffle(list_data_dict)  # Randomly shuffle the data for training

        rank0_print("Formatting inputs...Skip in lazy mode")
        processor = AutoProcessor.from_pretrained(processor_path)
        processor = update_processor_pixels(processor, min_pixels, max_pixels,
                                            video_max_pixels, video_min_pixels,
                                            video_max_frames, video_min_frames,
                                            video_fps)
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)
        self.list_data_dict = list_data_dict

        if data_packing:
            self.item_fn = self._get_packed_item
        else:
            self.item_fn = self._get_item

    def __len__(self):
        return len(self.list_data_dict)

    def load_data_list(self):
        pass

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = (
                cur_len if ("image" in sample) or ("video" in sample) else -cur_len
            )
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3
        num_final_retries = 30

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sources = self.list_data_dict[i]
                if isinstance(sources, dict):
                    sources = [sources]
                sample = self.item_fn(sources)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}{sources}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + attempt_idx, len(self.list_data_dict) - 1)
                sources = self.list_data_dict[next_index]
                if isinstance(sources, dict):
                    sources = [sources]

                sample = self.item_fn(sources)
                return sample
            except Exception as e:
                # no need to sleep
                print(
                    f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}{sources}. Exception:",
                    e,
                )
                pass

        try:
            sources = self.list_data_dict[i]
            if isinstance(sources, dict):
                sources = [sources]
            sample = self.item_fn(sources)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, sources) -> Dict[str, torch.Tensor]:
        # Counting the number of images
        source_data = sources[0] if isinstance(sources, list) else sources
        images_field = source_data.get("image", [])

        # Truncation policy (hybrid policy of tokens length & turns)
        if self.max_turns is not None or self.max_tokens is not None:
            sources[0] = _smart_truncate_conversations(
                sources[0],
                max_turns=self.max_turns,
                max_tokens=self.max_tokens,
                random_sampling=self.random_turn_sampling
            )
            # update source_data
            source_data = sources[0]
            images_field = source_data.get("image", [])

        if images_field:
            num_images = len(images_field) if isinstance(images_field, list) else (1 if images_field else 0)

            # Saving original parameters
            image_processor = self.processor.image_processor
            original_max_pixels = getattr(image_processor, 'max_pixels', None)
            original_size_longest_edge = None
            if hasattr(image_processor, 'size') and isinstance(image_processor.size, dict):
                original_size_longest_edge = image_processor.size.get('longest_edge', None)

            # Dynamically adjust the maximum number of pixels per image
            if num_images > 0 and hasattr(self, 'image_max_total_pixels'):
                # maximum number of pixels allowed for each image
                max_pixels_per_image = max(self.image_max_total_pixels // num_images, self.min_pixels)
                # adjust the processor Parameter
                if hasattr(image_processor, 'max_pixels'):
                    image_processor.max_pixels = min(image_processor.max_pixels, max_pixels_per_image)
                if hasattr(image_processor, 'size') and isinstance(image_processor.size, dict):
                    image_processor.size['longest_edge'] = min(image_processor.size['longest_edge'], max_pixels_per_image)

        data_dict = preprocess_qwen_visual(
            sources,
            self.processor,
            video_max_frames=self.video_max_frames,
        )
        
        if images_field:
            # Restoring the original parameter settings (to avoid affecting other samples)
            if original_max_pixels is not None and hasattr(image_processor, 'max_pixels'):
                image_processor.max_pixels = original_max_pixels
            if original_size_longest_edge is not None and hasattr(image_processor, 'size') and isinstance(image_processor.size, dict):
                image_processor.size['longest_edge'] = original_size_longest_edge

        seq_len = data_dict["input_ids"][0].size(0)

        if "image_grid_thw" in data_dict:
            grid_thw = data_dict.get("image_grid_thw")
            if not isinstance(grid_thw, Sequence):
                grid_thw = [grid_thw]
        else:
            grid_thw = None

        if "video_grid_thw" in data_dict:
            video_grid_thw = data_dict.get("video_grid_thw")
            if not isinstance(video_grid_thw, Sequence):
                video_grid_thw = [video_grid_thw]
            second_per_grid_ts = [2 / data_dict['metadata'][0][0].fps] * len(video_grid_thw)  # temporal_patch_size / fps
            # second_per_grid_ts = [1.0] * len(video_grid_thw) # hard code here as it is 2 / 2
        else:
            video_grid_thw = None
            second_per_grid_ts = None

        position_ids, _ = self.get_rope_index(
            self.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.cat(grid_thw, dim=0) if grid_thw else None,
            video_grid_thw=(
                torch.cat(video_grid_thw, dim=0) if video_grid_thw else None
            ),
            second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
        )

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [seq_len]
        data_dict["git_cfg"] = self.git_cfg

        text = self.processor.tokenizer.decode(
            data_dict["input_ids"][0], skip_special_tokens=False
        )

        labels = data_dict["labels"][0]
        labels = [
            tid if tid != -100 else self.processor.tokenizer.pad_token_id
            for tid in labels
        ]
        label = self.processor.tokenizer.decode(labels, skip_special_tokens=False)

        return data_dict

    def _get_packed_item(self, sources) -> Dict[str, torch.Tensor]:

        if isinstance(sources, dict):
            if isinstance(source, dict):
                sources = [sources]
            assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
            return self._get_item(sources)

        if isinstance(sources, list):
            data_list = []
            new_data_dict = {}
            for source in sources:
                if isinstance(source, dict):
                    source = [source]
                assert (
                    len(source) == 1
                ), f"Don't know why it is wrapped to a list.\n {source}"  # FIXME
                data_list.append(self._get_item(source))

            input_ids = torch.cat([d["input_ids"] for d in data_list], dim=1)
            labels = torch.cat([d["labels"] for d in data_list], dim=1)
            position_ids = torch.cat([d["position_ids"] for d in data_list], dim=2)
            attention_mask = [
                d["attention_mask"][0] for d in data_list if "attention_mask" in d
            ]
            new_data_dict = {
                "input_ids": input_ids,
                "labels": labels,
                "position_ids": position_ids,
                "attention_mask": attention_mask if attention_mask else None,
            }

            if any("pixel_values" in d for d in data_list):
                new_data_dict.update(
                    {
                        "pixel_values": torch.cat(
                            [
                                d["pixel_values"]
                                for d in data_list
                                if "pixel_values" in d
                            ],
                            dim=0,
                        ),
                        "image_grid_thw": torch.cat(
                            [
                                d["image_grid_thw"]
                                for d in data_list
                                if "image_grid_thw" in d
                            ],
                            dim=0,
                        ),
                    }
                )

            if any("pixel_values_videos" in d for d in data_list):
                new_data_dict.update(
                    {
                        "pixel_values_videos": torch.cat(
                            [
                                d["pixel_values_videos"]
                                for d in data_list
                                if "pixel_values_videos" in d
                            ],
                            dim=0,
                        ),
                        "video_grid_thw": torch.cat(
                            [
                                d["video_grid_thw"]
                                for d in data_list
                                if "video_grid_thw" in d
                            ],
                            dim=0,
                        ),
                    }
                )
            return new_data_dict


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, :, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids
        return batch


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids, attention_mask = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids", "attention_mask")
        )
        attention_mask = list(
            itertools.chain(
                *(
                    instance["attention_mask"]
                    for instance in instances
                    if "attention_mask" in instance
                )
            )
        )
        seq_lens = torch.tensor([0] + attention_mask, dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)
        position_ids = torch.cat(position_ids, dim=2)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw

        return batch


def make_supervised_data_module(processor, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = QwenLazySupervisedDataset(processor, data_args=data_args)
    if data_args.data_flatten or data_args.data_packing:
        data_collator = FlattenedDataCollatorForSupervisedDataset(processor.tokenizer)
        return dict(
            train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
        )
    data_collator = DataCollatorForSupervisedDataset(processor.tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


if __name__ == "__main__":
    pass