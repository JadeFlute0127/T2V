# tools/video_processor.py
import cv2
import base64
import numpy as np
import logging
import os
import json
import random

# 配置日志
logger = logging.getLogger(__name__)

def read_video(video_path: str, total_frames: int) -> tuple[list[str], float]:
    """
    读取视频并采样指定数量的关键帧，转换为Base64字符串列表
    :param video_path: 视频文件路径
    :param total_frames: 要采样的帧数
    :return: (选中的帧Base64列表, 视频时长)
    """
    # 打开视频
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"无法打开视频文件：{video_path}")

    try:
        # 初始化Base64帧列表
        base64_frames = []

        # 逐帧读取并转换为Base64
        while True:
            success, frame = video.read()
            if not success:
                break  # 无更多帧或读取失败

            # 编码为JPEG格式
            _, buffer = cv2.imencode('.jpg', frame)

            # 转换为Base64字符串
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            base64_frames.append(frame_base64)

        # 固定随机种子保证采样可复现
        random.seed(42)
        np.random.seed(42)

        # 处理视频帧数不足的情况
        frame_count = len(base64_frames)
        if frame_count == 0:
            raise ValueError(f"视频{video_path}无有效帧")
        if total_frames > frame_count:
            logger.warning(f"视频{video_path}总帧数({frame_count})不足采样数({total_frames})，仅返回所有帧")
            selected_indices = list(range(frame_count))
        else:
            # 均匀采样帧索引（参考原逻辑）
            if total_frames == 1:
                selected_indices = [np.random.choice(range(frame_count))]
            else:
                selected_indices = np.linspace(0, frame_count - 1, total_frames, dtype=int)

        # 提取选中的帧
        selected_base64_frames = [base64_frames[index] for index in selected_indices]

        # 计算视频时长
        fps = video.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps and fps > 0 else 0

        logger.info(f"视频{video_path}采样完成：总帧数={frame_count}，选中帧数={len(selected_base64_frames)}，时长={duration:.2f}秒")
        return selected_base64_frames, duration

    finally:
        # 释放视频资源
        video.release()

def prepare_base64frames(
        model_name: str,
        video_path: str,
        video_id: str,
        total_frames: int,
        video_tmp_dir: str = "./tmp"
) -> list[str]:
    """
    提取视频关键帧并保存到临时目录（带缓存机制），返回Base64字符串列表
    :param model_name: 模型名（用于日志，非核心）
    :param video_path: 视频文件路径
    :param video_id: 视频唯一标识（如sample_id）
    :param total_frames: 要采样的帧数
    :param video_tmp_dir: 临时文件根目录
    :return: 关键帧Base64字符串列表
    """
    # 1. 定义临时目录和文件路径（严格参考原命名逻辑）
    image_subdir = os.path.join(video_tmp_dir, model_name, video_id, f"{total_frames}_frames")
    tmp_base64_file = os.path.join(image_subdir, "base64frames.json")

    # 2. 检查缓存：若临时目录存在，优先读取缓存
    if os.path.exists(image_subdir) and len(os.listdir(image_subdir)) != 0:
        logger.info(f"【模型{model_name} 视频{video_id}】检测到临时缓存目录，读取已有Base64帧")

        # 读取base64frames.json
        if os.path.exists(tmp_base64_file):
            try:
                with open(tmp_base64_file, "r", encoding="utf-8") as f:
                    base64frames = json.load(f)
                logger.info(f"【模型{model_name} 视频{video_id}】从缓存读取{len(base64frames)}个Base64帧")
                return base64frames
            except Exception as e:
                logger.error(f"【模型{model_name} 视频{video_id}】读取缓存文件失败：{e}，重新提取帧")

        # 备用方案：从jpg文件重新生成Base64
        logger.info(f"【模型{model_name} 视频{video_id}】缓存文件损坏，从jpg文件重建Base64帧")
        base64frames = []
        for i in range(total_frames):
            jpg_path = os.path.join(image_subdir, f"frame_{i}.jpg")
            if not os.path.exists(jpg_path):
                logger.warning(f"【模型{model_name} 视频{video_id}】缺失帧文件{jpg_path}，跳过")
                continue
            with open(jpg_path, "rb") as f:
                frame_base64 = base64.b64encode(f.read()).decode('utf-8')
                base64frames.append(frame_base64)
        return base64frames

    # 3. 无缓存：提取帧并保存到临时目录
    logger.info(f"【模型{model_name} 视频{video_id}】无缓存，开始提取并保存关键帧")
    try:
        # 创建临时目录
        os.makedirs(image_subdir, exist_ok=True)

        # 提取帧的Base64列表
        base64frames, _ = read_video(video_path, total_frames)

        # 保存每个帧为jpg文件（参考原逻辑）
        for i, frame_base64 in enumerate(base64frames):
            jpg_path = os.path.join(image_subdir, f"frame_{i}.jpg")
            with open(jpg_path, "wb") as f:
                f.write(base64.b64decode(frame_base64))

        # 保存Base64列表到json文件
        with open(tmp_base64_file, "w", encoding="utf-8") as f:
            json.dump(base64frames, f, ensure_ascii=False)

        logger.info(f"【模型{model_name} 视频{video_id}】成功保存{len(base64frames)}个帧到临时目录：{image_subdir}")
        return base64frames

    except Exception as e:
        logger.error(f"【模型{model_name} 视频{video_id}】提取/保存帧失败：{e}")
        return []