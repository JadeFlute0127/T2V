import os
import logging
import shutil
from typing import List, Dict

# ===================== 配置区域 =====================
# 待处理的模型名列表（与evaluation.py中的TARGET_MODEL_NAME保持一致）
TARGET_MODEL_NAME = [
    # "Veo3_1_fast",
    # "Sora",
    # "LongCat-Video",
    # "HunyuanVideo-1.5",
    # "LTX-2",
    # "FastWan2.2-TI2V-5B-FullAttn-Diffusers",
    # "Sora2",
    "Veo3_1"
]

# 视频文件根目录（与evaluation.py中的VIDEO_BASE_DIR保持一致）
VIDEO_BASE_DIR = "input/dataset/video"

# 子目录名（固定为Engineering）
SUB_DIR = "patch"

# 文件名替换规则：前缀替换（733d00 → 733d11）
OLD_PREFIX = "733d0061a45c022ad410ff6af7356d4434843"
NEW_PREFIX = "733d1161a45c022ad410ff6af7356d4434843"

# 文件名后缀列表（对应需要替换的7个文件）
FILE_SUFFIXES = [
    "75e.mp4",
    "76e.mp4",
    "77e.mp4",
    "78e.mp4",
    "79e.mp4",
    "80e.mp4",
    "81e.mp4"
]

# 新增：需要处理的子目录列表（移动文件+删除空目录）
TARGET_SUBDIRS = [
    "Natural_Science",
    "Human",
    "Healthcare",
    "Engineering",
    "patch"
]
# ====================================================

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def generate_file_mapping() -> Dict[str, str]:
    """生成源文件名→目标文件名的映射字典"""
    file_mapping = {}
    for suffix in FILE_SUFFIXES:
        old_filename = f"{OLD_PREFIX}{suffix}"
        new_filename = f"{NEW_PREFIX}{suffix}"
        file_mapping[old_filename] = new_filename
    return file_mapping

# 新增：将子目录中的视频文件移动到模型根目录
def move_videos_to_parent_dir(model_name: str):
    """
    将指定模型下TARGET_SUBDIRS中的所有mp4文件移动到模型根目录
    :param model_name: 模型名
    """
    model_root_dir = os.path.join(VIDEO_BASE_DIR, model_name)
    if not os.path.exists(model_root_dir):
        logger.warning(f"模型{model_name}根目录不存在，跳过文件移动：{model_root_dir}")
        return

    total_moved = 0
    total_move_failed = 0

    logger.info(f"\n开始移动模型{model_name}的视频文件到根目录：{model_root_dir}")
    for subdir in TARGET_SUBDIRS:
        subdir_path = os.path.join(model_root_dir, subdir)
        if not os.path.exists(subdir_path):
            logger.debug(f"子目录不存在，跳过：{subdir_path}")
            continue

        # 遍历子目录中的所有mp4文件
        for filename in os.listdir(subdir_path):
            if not filename.lower().endswith(".mp4"):
                continue

            src_path = os.path.join(subdir_path, filename)
            dst_path = os.path.join(model_root_dir, filename)

            # 处理目标文件已存在的情况（添加后缀避免覆盖）
            counter = 1
            while os.path.exists(dst_path):
                name, ext = os.path.splitext(filename)
                dst_path = os.path.join(model_root_dir, f"{name}_{counter}{ext}")
                counter += 1

            try:
                # 移动文件
                shutil.move(src_path, dst_path)
                logger.info(f"文件移动成功：{src_path} → {dst_path}")
                total_moved += 1
            except PermissionError:
                logger.error(f"权限不足，无法移动文件：{src_path}")
                total_move_failed += 1
            except OSError as e:
                logger.error(f"系统错误，移动文件失败：{src_path} | 错误：{str(e)}")
                total_move_failed += 1
            except Exception as e:
                logger.error(f"未知错误，移动文件失败：{src_path} | 错误：{str(e)}")
                total_move_failed += 1

    logger.info(f"模型{model_name}文件移动完成：成功{total_moved}个，失败{total_move_failed}个")

# 新增：删除空的子目录
def delete_empty_subdirs(model_name: str):
    """
    删除指定模型下TARGET_SUBDIRS中的空目录
    :param model_name: 模型名
    """
    model_root_dir = os.path.join(VIDEO_BASE_DIR, model_name)
    if not os.path.exists(model_root_dir):
        logger.warning(f"模型{model_name}根目录不存在，跳过空目录删除：{model_root_dir}")
        return

    total_deleted = 0
    total_delete_failed = 0

    logger.info(f"\n开始删除模型{model_name}的空子目录")
    for subdir in TARGET_SUBDIRS:
        subdir_path = os.path.join(model_root_dir, subdir)
        if not os.path.exists(subdir_path):
            continue

        # 检查目录是否为空
        try:
            if not os.listdir(subdir_path):
                os.rmdir(subdir_path)
                logger.info(f"空目录删除成功：{subdir_path}")
                total_deleted += 1
            else:
                logger.warning(f"目录非空，跳过删除：{subdir_path}")
        except PermissionError:
            logger.error(f"权限不足，无法删除目录：{subdir_path}")
            total_delete_failed += 1
        except OSError as e:
            logger.error(f"系统错误，删除目录失败：{subdir_path} | 错误：{str(e)}")
            total_delete_failed += 1
        except Exception as e:
            logger.error(f"未知错误，删除目录失败：{subdir_path} | 错误：{str(e)}")
            total_delete_failed += 1

    logger.info(f"模型{model_name}空目录删除完成：成功{total_deleted}个，失败{total_delete_failed}个")

def replace_video_files():
    """核心逻辑：检查并替换指定目录下的视频文件"""
    # 生成文件映射
    file_mapping = generate_file_mapping()
    logger.info(f"生成文件替换映射：共{len(file_mapping)}个文件待检查")
    for old, new in file_mapping.items():
        logger.debug(f"  {old} → {new}")

    # 统计变量
    total_checked = 0
    total_replaced = 0
    total_skipped = 0
    total_failed = 0

    # 遍历每个模型
    for model_name in TARGET_MODEL_NAME:
        logger.info(f"\n{'='*40} 处理模型：{model_name} {'='*40}")

        # 构造完整目录路径
        target_dir = os.path.join(VIDEO_BASE_DIR, model_name, SUB_DIR)
        logger.info(f"检查目录：{os.path.abspath(target_dir)}")

        # 检查目录是否存在
        if not os.path.exists(target_dir):
            logger.error(f"模型{model_name}的{SUB_DIR}目录不存在，跳过")
            target_dir = os.path.join(VIDEO_BASE_DIR, model_name)

        # 遍历每个待替换文件
        for old_filename, new_filename in file_mapping.items():
            total_checked += 1
            old_file_path = os.path.join(target_dir, old_filename)
            new_file_path = os.path.join(target_dir, new_filename)

            # 检查源文件是否存在
            if not os.path.exists(old_file_path):
                logger.warning(f"源文件不存在，跳过：{old_file_path}")
                total_skipped += 1
                continue

            try:
                # 检查目标文件是否已存在
                if os.path.exists(new_file_path):
                    logger.warning(f"目标文件已存在，将覆盖：{new_file_path}")
                    # 可选：如果不想覆盖，取消下面注释并跳过
                    # total_skipped += 1
                    # continue

                # 执行文件重命名（替换）
                os.rename(old_file_path, new_file_path)
                logger.info(f"文件替换成功：{old_filename} → {new_filename}")
                total_replaced += 1

            except PermissionError:
                logger.error(f"权限不足，无法替换文件：{old_file_path}")
                total_failed += 1
            except OSError as e:
                logger.error(f"系统错误，替换文件失败：{old_file_path} | 错误：{str(e)}")
                total_failed += 1
            except Exception as e:
                logger.error(f"未知错误，替换文件失败：{old_file_path} | 错误：{str(e)}")
                total_failed += 1

    # 输出统计结果
    logger.info(f"\n{'='*60}")
    logger.info(f"文件替换任务完成！")
    logger.info(f"总计检查文件数：{total_checked}")
    logger.info(f"成功替换数：{total_replaced}")
    logger.info(f"跳过数（文件不存在/已存在）：{total_skipped}")
    logger.info(f"失败数：{total_failed}")
    logger.info(f"{'='*60}")

    # 新增：执行文件移动和空目录删除
    logger.info(f"\n{'='*60}")
    logger.info(f"开始执行视频文件移动和空目录删除任务")
    logger.info(f"{'='*60}")

    for model_name in TARGET_MODEL_NAME:
        # 移动子目录视频到模型根目录
        move_videos_to_parent_dir(model_name)
        # 删除空的子目录
        delete_empty_subdirs(model_name)

if __name__ == "__main__":
    try:
        logger.info("开始执行视频文件替换脚本")
        replace_video_files()
    except Exception as e:
        logger.error(f"脚本执行失败：{str(e)}", exc_info=True)
        exit(1)