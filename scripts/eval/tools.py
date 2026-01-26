import cv2
import base64
import os
from PIL import Image
import tempfile

def extract_video_frames(video_path, frame_numbers=[0, 5, -1]):
    """
    从视频中提取指定帧（默认首帧、第5帧、末帧），保存为临时图片并返回Base64编码列表
    :param video_path: 本地视频文件路径
    :param frame_numbers: 要提取的帧序号（0=首帧，-1=末帧）
    :return: 帧的Base64编码列表 + 帧的序号说明
    """
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"无法打开视频文件：{video_path}")

    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_base64_list = []
    frame_info_list = []

    for frame_idx in frame_numbers:
        # 处理末帧
        if frame_idx == -1:
            frame_idx = total_frames - 1
        # 确保帧序号有效
        frame_idx = max(0, min(frame_idx, total_frames - 1))

        # 定位到目标帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # 转换颜色空间（OpenCV默认BGR，PIL是RGB）
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 保存为临时图片
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_img:
            img = Image.fromarray(frame_rgb)
            img.save(temp_img, format="JPEG")
            temp_img_path = temp_img.name

        # 读取图片并转Base64
        with open(temp_img_path, "rb") as f:
            frame_base64 = base64.b64encode(f.read()).decode("utf-8")

        # 清理临时文件
        os.unlink(temp_img_path)

        frame_base64_list.append(frame_base64)
        frame_info_list.append(f"第{frame_idx+1}帧（总帧数：{total_frames}）")

    cap.release()
    return frame_base64_list, frame_info_list