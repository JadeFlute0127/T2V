import cv2
import base64
import numpy as np
import logging
import os
import json
import random

# 配置日志
logger = logging.getLogger(__name__)


def read_video_dynamic(video_path: str, total_frames: int) -> tuple[list[str], float]:
    """
    读取视频并按【每秒2帧】采样关键帧，转换为Base64字符串列表
    适配8/10/14/15秒等不同时长视频，忽略外部传入的total_frames参数（兼容调用）
    :param video_path: 视频文件路径
    :param total_frames: 外部传入参数（已忽略，仅做兼容）
    :return: (选中的帧Base64列表, 视频时长)
    """
    # 打开视频
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"无法打开视频文件：{video_path}")

    try:
        # 初始化Base64帧列表
        base64_frames = []

        # 逐帧读取并转换为Base64（保留原逻辑，读取所有帧）
        while True:
            success, frame = video.read()
            if not success:
                break  # 无更多帧或读取失败

            # 编码为JPEG格式
            _, buffer = cv2.imencode('.jpg', frame)
            # 转换为Base64字符串
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            base64_frames.append(frame_base64)

        # 基础参数计算
        frame_count = len(base64_frames)  # 视频实际总帧数
        if frame_count == 0:
            raise ValueError(f"视频{video_path}无有效帧")
        fps = video.get(cv2.CAP_PROP_FPS)  # 视频实际帧率
        # 计算视频实际时长（秒），处理fps为0/无效的情况
        if not fps or fps <= 0:
            fps = 25  # 兜底默认帧率，避免除零错误
            logger.warning(f"视频{video_path}帧率获取失败，使用兜底帧率{fps}fps")
        video_duration = frame_count / fps  # 核心：视频实际时长
        logger.info(f"视频{video_path}基础信息：总帧数={frame_count}，帧率={fps:.2f}fps，实际时长={video_duration:.2f}秒")

        # ========== 核心修改：每秒2帧，动态计算采样逻辑 ==========
        sample_fps = 2  # 固定：每秒采样2帧
        time_interval = 1 / sample_fps  # 采样时间间隔：0.5秒/帧
        target_sample_num = int(video_duration * sample_fps)  # 动态总采样数=时长×2
        logger.info(f"按{sample_fps}帧/秒采样：目标采样数={target_sample_num}，采样时间间隔={time_interval}秒")

        # 生成采样时间点（0, 0.5, 1.0, 1.5, ..., 视频时长）
        sample_times = np.arange(0, video_duration, time_interval)
        # 将时间点转换为帧索引（时间×帧率），并做边界限制（0 ~ frame_count-1）
        sample_indices = np.clip(np.round(sample_times * fps).astype(int), 0, frame_count - 1)
        # 去重（避免浮点计算/帧率误差导致同一帧被多次采样）
        sample_indices = np.unique(sample_indices)
        # 最终选中的帧索引（保证采样数不超过实际总帧数）
        selected_indices = sample_indices[:target_sample_num]
        # ========================================================

        # 提取选中的帧
        selected_base64_frames = [base64_frames[index] for index in selected_indices]

        logger.info(
            f"视频{video_path}采样完成：实际采样数={len(selected_base64_frames)}，目标采样数={target_sample_num}，时长={video_duration:.2f}秒")
        return selected_base64_frames, video_duration

    finally:
        # 释放视频资源
        video.release()


def prepare_base64frames_dynamic(
        model_name: str,
        video_path: str,
        video_id: str,
        total_frames: int = 0,
        video_tmp_dir: str = "./tmp"
) -> list[str]:
    """
    提取视频关键帧并保存到临时目录（带缓存机制），返回Base64字符串列表
    :param model_name: 模型名（用于日志，非核心）
    :param video_path: 视频文件路径
    :param video_id: 视频唯一标识（如sample_id）
    :param total_frames: 要采样的帧数（外部传入，内部read_video已忽略）
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

        # 提取帧的Base64列表（调用修改后的read_video，按每秒2帧采样）
        base64frames, _ = read_video_dynamic(video_path, total_frames)

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
