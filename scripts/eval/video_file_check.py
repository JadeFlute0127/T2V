import os
import json
import logging
from typing import Dict, List, Set

# ===================== 配置区域 =====================
# 核心路径配置（可根据实际情况调整）
SAMPLE_JSONL_PATH = "input/sample.jsonl"  # 样本ID文件路径
VIDEO_BASE_DIR = "input/dataset/video"            # 模型视频根目录
OUTPUT_REPORT_PATH = "missing_video_analysis_report.json"  # 分析报告输出路径
# ====================================================

# 配置日志（控制台+文件输出，便于排查问题）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # 控制台输出
        logging.FileHandler("video_missing_analysis.log", encoding="utf-8")  # 日志文件
    ]
)
logger = logging.getLogger(__name__)

def read_sample_ids() -> Set[str]:
    """
    第一步：读取sample.jsonl，解析所有唯一的sample_id
    返回：去重后的sample_id集合（便于快速对比）
    """
    sample_ids = set()

    # 检查文件是否存在
    if not os.path.exists(SAMPLE_JSONL_PATH):
        logger.error(f"样本文件不存在：{SAMPLE_JSONL_PATH}")
        return sample_ids

    logger.info(f"开始读取样本文件：{SAMPLE_JSONL_PATH}")
    with open(SAMPLE_JSONL_PATH, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue  # 跳过空行

            try:
                # 解析JSON行，提取id字段
                sample_data = json.loads(line)
                sample_id = sample_data.get("id")
                if sample_id is None:
                    logger.warning(f"第{line_num}行：缺少id字段，跳过")
                    continue

                # 统一转为字符串（避免数字/字符串类型不一致）
                sample_id_str = str(sample_id).strip()
                sample_ids.add(sample_id_str)
            except json.JSONDecodeError as e:
                logger.error(f"第{line_num}行：JSON解析失败 - {e}，跳过")
            except Exception as e:
                logger.error(f"第{line_num}行：处理异常 - {e}，跳过")

    logger.info(f"样本文件读取完成，共解析到 {len(sample_ids)} 个唯一样本ID")
    return sample_ids

def collect_model_video_ids() -> Dict[str, Set[str]]:
    """
    第二步：遍历VIDEO_BASE_DIR下的所有model目录，收集每个模型的视频ID
    返回：{model_name: 视频ID集合}
    """
    model_video_map = {}

    # 检查视频根目录是否存在
    if not os.path.exists(VIDEO_BASE_DIR):
        logger.error(f"视频根目录不存在：{VIDEO_BASE_DIR}")
        return model_video_map

    logger.info(f"开始遍历视频目录：{VIDEO_BASE_DIR}")
    # 遍历所有子目录（每个子目录对应一个model_name）
    for entry in os.scandir(VIDEO_BASE_DIR):
        if not entry.is_dir():
            continue  # 跳过文件，只处理目录

        model_name = entry.name
        if model_name == ".cache":
            continue
        video_ids = set()
        logger.info(f"  处理模型目录：{model_name}")

        # 遍历该模型目录下的所有文件
        try:
            for file_entry in os.scandir(entry.path):
                if not file_entry.is_file():
                    continue

                filename = file_entry.name
                # 只处理.mp4文件，且文件名格式为 {id}.mp4
                if filename.lower().endswith(".mp4"):
                    # 提取视频ID（去掉.mp4后缀）
                    video_id = os.path.splitext(filename)[0].strip()
                    video_ids.add(video_id)
                    logger.debug(f"    找到视频：{filename} -> ID: {video_id}")
        except PermissionError:
            logger.error(f"  无权限访问模型目录：{model_name}，跳过")
            continue
        except Exception as e:
            logger.error(f"  遍历模型目录 {model_name} 异常 - {e}，跳过")

        # 保存该模型的视频ID集合
        model_video_map[model_name] = video_ids
        logger.info(f"  模型 {model_name} 共找到 {len(video_ids)} 个视频ID")

    logger.info(f"视频目录遍历完成，共识别到 {len(model_video_map)} 个模型")
    return model_video_map

def analyze_missing_ids(sample_ids: Set[str], model_video_map: Dict[str, Set[str]]) -> Dict[str, List[str]]:
    """
    第三步：对比样本ID和各模型视频ID，分析缺失的ID
    返回：{model_name: [缺失的ID列表]}
    """
    missing_ids_map = {}
    logger.info("开始分析各模型缺失的视频ID")

    if not sample_ids:
        logger.error("无样本ID可对比，分析终止")
        return missing_ids_map

    for model_name, video_ids in model_video_map.items():
        # 计算样本ID中不在该模型视频ID里的ID（按字母/数字排序，便于查看）
        missing_ids = sorted([sid for sid in sample_ids if sid not in video_ids])
        missing_ids_map[model_name] = missing_ids

        # 日志输出该模型的缺失情况
        if missing_ids:
            logger.warning(f"模型 {model_name} 缺失 {len(missing_ids)} 个视频ID：{missing_ids}")
        else:
            logger.info(f"模型 {model_name} 包含所有样本ID，无缺失")

    return missing_ids_map

def generate_analysis_report(missing_ids_map: Dict[str, List[str]], sample_ids: Set[str]):
    """
    生成可视化的分析报告（控制台输出+JSON文件保存）
    """
    # 1. 控制台输出汇总报告
    logger.info("\n" + "="*80)
    logger.info("视频ID缺失分析报告")
    logger.info("="*80)

    total_models = len(missing_ids_map)
    total_sample_ids = len(sample_ids)
    logger.info(f"样本总ID数：{total_sample_ids}")
    logger.info(f"模型总数：{total_models}")
    logger.info("\n各模型缺失详情：")

    for model_name, missing_ids in missing_ids_map.items():
        missing_count = len(missing_ids)
        completion_rate = (total_sample_ids - missing_count) / total_sample_ids * 100 if total_sample_ids > 0 else 0
        logger.info(f"  模型 {model_name}：")
        logger.info(f"    - 缺失ID数：{missing_count}")
        logger.info(f"    - 完成率：{completion_rate:.2f}%")
        if missing_ids:
            logger.info(f"    - 缺失ID列表：{missing_ids}")
        logger.info("    ---")

    # 2. 保存JSON格式报告（便于后续处理）
    report_data = {
        "summary": {
            "total_sample_ids": total_sample_ids,
            "total_models": total_models,
            "analysis_time": os.popen("date").read().strip()  # 简单的时间戳
        },
        "model_details": missing_ids_map
    }

    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    logger.info(f"\n分析报告已保存至：{OUTPUT_REPORT_PATH}")
    logger.info("="*80)

def main():
    """主函数：执行完整的分析流程"""
    try:
        # 第一步：读取样本ID
        sample_ids = read_sample_ids()
        if not sample_ids:
            logger.error("未解析到任何样本ID，程序退出")
            return

        # 第二步：收集各模型视频ID
        model_video_map = collect_model_video_ids()
        if not model_video_map:
            logger.error("未识别到任何模型目录，程序退出")
            return

        # 第三步：分析缺失ID
        missing_ids_map = analyze_missing_ids(sample_ids, model_video_map)

        # 生成分析报告
        generate_analysis_report(missing_ids_map, sample_ids)

        logger.info("分析流程执行完成！")

    except Exception as e:
        logger.error(f"程序执行异常：{e}", exc_info=True)

if __name__ == "__main__":
    main()