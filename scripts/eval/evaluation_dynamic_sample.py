import os
import json
import random
import logging
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import AzureOpenAI, RateLimitError, APIError, Timeout
import time
import tools
import os
import json
from tools.video_dynamic_processor import prepare_base64frames_dynamic

# ===================== 配置区域（仅修改DEFAULT_SAMPLE_FRAMES注释，其他无改动） =====================
# 建议通过环境变量设置 API Key，避免硬编码
CONTROL_NUM = 1  # 每次处理的最大数据量
MAX_WORKERS = 5  # 降低并发数，避免Azure API限流（原500过高）
BASE_RATE = 200
JSONL_DATA_DIR = "input/dataset"  # 第一步：jsonl文件根目录
VIDEO_BASE_DIR = "input/dataset/video"  # 第二步：视频文件根目录（下级为model_name）
OUTPUT_DIR = "output"  # 输出目录
API_RETRY_TIMES = 3  # API 调用失败重试次数
LANGUAGE = 'en'
MAX_TOKENS = 16384  # 适配模型的最大token限制
ENABLED_SHUFFLE = True
ENABLED_SAMPLE_TEST = True
# 修复：sora2后补充逗号，修正列表语法错误
TARGET_MODEL_NAME = [
    # "Sora",
    # "Veo3_1-fast",
    # "LTX2",
    # "FastWan2_2-5B",
    # "Sora2",
    # "Veo3_1",
    # "LongCat-Video",
    # "HunyuanVideo-1.5",
    # "Wan2_6",
    # "Kling2_6"
]

# 新增：每个模型失败样本的最大重试次数
MODEL_ERROR_RETRY_TIMES = 10

evaluation_mapping = {
    "pg_specification": "Prompt Grounding: This dimension focuses on whether the video accurately displays the scene described in the prompt.",
    "scc_specification": "Scientific and Causal Correctness: This dimension focuses on whether the video displays the expected correct phenomena/images, and whether the causal logic is self-consistent and reasonable.",
    "sc_specification": "Spatiotemporal Consistency: This dimension focuses on whether the causal relationships are reasonable across time, space, and spatiotemporal relationships.",
    "lpf_specification": "Low-level Perceptual Fidelity: This dimension assesses the low-level perceptual quality of the synthesized video, including temporal coherence, motion dynamics, and the visual quality of individual frames.",
}
VIDEO_TMP_DIR = "tmp/video_frames"  # 视频帧临时缓存目录
# #################### 核心适配修改 ####################
# 保留15仅作为【缓存目录命名占位符】，实际采样逻辑由video_processor.py按「每秒2帧」动态计算
# 不修改该值，确保缓存目录命名兼容（{total_frames}_frames），原有缓存无需重新生成
DEFAULT_SAMPLE_FRAMES = 0   # 动态采样，每秒2帧
# ######################################################
# ====================================================

# ===================== 日志配置修改 =====================
# 创建日志目录
LOG_DIR = os.path.join(OUTPUT_DIR, "log")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIDEO_TMP_DIR, exist_ok=True)

# 移除原有basicConfig，手动配置日志（支持多模型独立日志文件）
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers.clear()  # 清空默认处理器

# 1. 控制台处理器（保留原有输出）
console_handler = logging.StreamHandler()
console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# 2. 模型日志处理器缓存（key: model_name, value: FileHandler）
model_log_handlers = {}
# 日志文件锁（保证多线程下日志文件创建/写入安全）
log_file_lock = threading.Lock()
# ====================================================

# Azure OpenAI 客户端初始化
client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint="https://jakeh-mif61zia-eastus2.cognitiveservices.azure.com/",
    api_key="",
)

# 线程锁（保证并发下单个model_name的jsonl文件追加写入安全）
file_append_lock = threading.Lock()

# ===================== 新增核心函数：读取已处理的样本ID =====================
def get_processed_sample_ids(model_name: str) -> set:
    """
    读取指定模型的evaluation.jsonl文件，提取已处理的样本ID
    :param model_name: 模型名
    :return: 已处理的样本ID集合（字符串类型）
    """
    processed_ids = set()
    output_filename = f"{model_name}-evaluation.jsonl"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    # 文件不存在则返回空集合（无已处理样本）
    if not os.path.exists(output_path):
        logger.info(f"模型{model_name}的结果文件不存在（{output_path}），无已处理样本")
        return processed_ids

    logger.info(f"开始读取模型{model_name}的已处理样本ID：{output_path}")
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    # 解析每行JSON，提取id字段
                    result_data = json.loads(line)
                    sample_id = result_data.get("id")
                    if sample_id is None:
                        logger.warning(f"模型{model_name}结果文件第{line_num}行：缺少id字段，跳过")
                        continue

                    # 统一转为字符串，避免类型不一致导致的对比错误
                    sample_id_str = str(sample_id).strip()
                    processed_ids.add(sample_id_str)
                except json.JSONDecodeError as e:
                    logger.error(f"模型{model_name}结果文件第{line_num}行：JSON解析失败 - {e}，跳过")
                except Exception as e:
                    logger.error(f"模型{model_name}结果文件第{line_num}行：处理异常 - {e}，跳过")

        logger.info(f"模型{model_name}已处理样本ID读取完成，共{len(processed_ids)}个")
    except PermissionError:
        logger.error(f"无权限读取模型{model_name}的结果文件：{output_path}")
    except Exception as e:
        logger.error(f"读取模型{model_name}已处理样本ID失败：{str(e)}")

    return processed_ids

# ====================================================

# 日志修改：为指定模型添加独立的日志文件处理器
def add_model_log_handler(model_name: str):
    """
    为指定模型创建并添加日志文件处理器
    :param model_name: 模型名
    """
    with log_file_lock:
        if model_name in model_log_handlers:
            return  # 已存在则直接返回

        # 生成日志文件名：model_name-YYYYMMDD_HHMMSS.log
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{model_name}-{timestamp}.log"
        log_filepath = os.path.join(LOG_DIR, log_filename)

        # 创建文件处理器
        file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)

        # 添加处理器并缓存
        logger.addHandler(file_handler)
        model_log_handlers[model_name] = file_handler
        logger.info(f"为模型{model_name}创建日志文件：{log_filepath}")

# 日志修改：移除指定模型的日志文件处理器
def remove_model_log_handler(model_name: str):
    """
    移除指定模型的日志文件处理器（避免文件句柄泄露）
    :param model_name: 模型名
    """
    with log_file_lock:
        if model_name not in model_log_handlers:
            return

        # 移除处理器并关闭文件
        handler = model_log_handlers.pop(model_name)
        logger.removeHandler(handler)
        handler.close()
        logger.info(f"已关闭模型{model_name}的日志文件处理器")

def chat_gpt(prompt: str, dimension: str, base64_images: []) -> str:
    """
    调用 ChatGPT 接口（适配单个维度调用），处理重试和速率限制
    :param prompt: 输入的提示词（单个维度）
    :param dimension: 当前处理的维度名称（日志用）
    :return: GPT 返回的响应文本
    """

    content = [{"type": "text", "text": prompt}]
    for base64_image in base64_images:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        )
    for retry in range(API_RETRY_TIMES):
        try:
            response = client.chat.completions.create(
                model="gpt-5.2-chat",
                # messages=[
                #     {"role": "user", "content": prompt},
                # ],
                messages=[
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                stream=False,
                timeout=30,
                max_completion_tokens=MAX_TOKENS,
            )
            # 增加微小延迟，避免高频调用触发限流
            time.sleep(0.5)
            return response.choices[0].message.content
        except (RateLimitError, APIError, Timeout) as e:
            logger.error(
                f"【维度{dimension}】API调用异常: {type(e).__name__} - {e}，重试次数：{retry + 1}/{API_RETRY_TIMES}")
            # 限流时增加延迟
            time.sleep(2 * (retry + 1))
        except Exception as e:
            logger.error(f"【维度{dimension}】未知错误: {e}，重试次数：{retry + 1}/{API_RETRY_TIMES}")
            # time.sleep(1 * (retry + 1))

    raise Exception(f"【维度{dimension}】API 调用失败，已重试 {API_RETRY_TIMES} 次")


def get_prompt_template_for_dimension() -> str:
    """
    读取单个维度评估的prompt模板（需提前在input目录创建）
    模板需包含 {dimension}、{dimension_content}、{video_sequence}（en）/{video_path}（cn） 占位符
    """
    template_filename = "instruction_cn.txt" if LANGUAGE == 'cn' else "instruction_en.txt"
    TEMPLATE_PATH = os.path.join("input/template", template_filename)

    try:
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
            template = f.read()
        logger.info(f"成功读取维度评估prompt模板（{LANGUAGE}）")
        return template
    except FileNotFoundError:
        raise Exception(f"维度评估模板文件不存在：{TEMPLATE_PATH}")
    except Exception as e:
        raise Exception(f"读取维度评估模板失败：{str(e)}")


def load_all_jsonl_files() -> dict:
    """
    第一步：读取/input/dataset目录下的所有jsonl文件，按id聚合数据
    :return: 字典 {id: evaluation_specification}
    """
    jsonl_data = {}
    if not os.path.exists(JSONL_DATA_DIR):
        raise Exception(f"JSONL数据源目录不存在：{JSONL_DATA_DIR}")

    # 遍历目录下所有jsonl文件
    for filename in os.listdir(JSONL_DATA_DIR):
        if ENABLED_SAMPLE_TEST and filename != "sample.jsonl":
            continue
        if not filename.endswith(".jsonl"):
            continue
        file_path = os.path.join(JSONL_DATA_DIR, filename)
        logger.info(f"开始读取JSONL文件：{file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # 校验必要字段
                    if "id" not in data:
                        logger.warning(f"{file_path}第{line_num}行：缺少id字段，跳过")
                        continue
                    id = data["id"].strip()
                    # 构造评估spec字典（兼容字段缺失）
                    spec_dict = {
                        "pg_specification": data.get("pg_specification", ""),
                        "scc_specification": data.get("scc_specification", ""),
                        "sc_specification": data.get("sc_specification", ""),
                        "lpf_specification": data.get("lpf_specification", ""),
                        "high_level_evaluation_guide": data.get("high_level_evaluation_guide", "")
                    }
                    if id in jsonl_data:
                        logger.warning(f"{file_path}第{line_num}行：id={id}已存在，覆盖原有数据")
                    jsonl_data[id] = spec_dict
                except json.JSONDecodeError as e:
                    logger.error(f"{file_path}第{line_num}行：JSON解析失败 - {e}，跳过")
                except Exception as e:
                    logger.error(f"{file_path}第{line_num}行：处理失败 - {e}，跳过")

    logger.info(f"共读取 {len(jsonl_data)} 条有效样本（id-evaluation_specification映射）")
    return jsonl_data


def get_video_path_by_id(model_name: str, sample_id: str) -> str:
    """
    第二步：获取指定model_name下对应id的视频文件路径
    :param model_name: 模型名（对应VIDEO_BASE_DIR下的子目录）
    :param sample_id: 样本id
    :return: 视频文件路径，不存在则抛出异常
    """
    video_dir = os.path.join(VIDEO_BASE_DIR, model_name)
    # 先检查模型目录是否存在
    if not os.path.exists(video_dir):
        raise Exception(f"模型视频目录不存在：{video_dir}")
    video_path = os.path.join(video_dir, f"{sample_id}.mp4")
    if not os.path.exists(video_path):
        raise Exception(f"视频文件不存在：{video_path}")
    return video_path


def get_video_key_frames_base64(
        model_name: str,
        sample_id: str,
        total_frames: int = DEFAULT_SAMPLE_FRAMES,
        video_tmp_dir: str = VIDEO_TMP_DIR
) -> list[str]:
    """
    获取指定模型+样本ID的视频关键帧Base64列表（带缓存）
    :param model_name: 模型名
    :param sample_id: 样本ID（作为video_id）
    :param total_frames: 仅为缓存目录命名用，实际由video_processor.py按每秒2帧动态采样
    :param video_tmp_dir: 临时缓存目录
    :return: Base64字符串列表，失败则抛出异常
    """
    try:
        # 1. 获取视频路径
        video_path = get_video_path_by_id(model_name, sample_id)

        # 2. 调用工具函数（带缓存，total_frames仅用于缓存目录命名）
        base64frames = prepare_base64frames_dynamic(
            model_name=model_name,
            video_path=video_path,
            video_id=sample_id,  # 使用sample_id作为视频唯一标识
            total_frames=total_frames,
            video_tmp_dir=video_tmp_dir
        )

        # 3. 校验结果
        if not base64frames:
            raise Exception(f"视频{video_path}未提取到有效关键帧")

        logger.info(f"【模型{model_name} 样本{sample_id}】最终获取{len(base64frames)}个关键帧Base64字符串（每秒2帧动态采样）")
        return base64frames

    except Exception as e:
        error_msg = f"【模型{model_name} 样本{sample_id}】提取关键帧失败：{str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)


def parse_gpt_response_for_dimension(response_text: str, dimension: str) -> dict:
    """
    第四步：解析GPT返回的单个维度响应（严格对齐instruction.txt定义的JSON格式）
    :param response_text: GPT响应文本
    :param dimension: 当前维度名称
    :return: 解析后的维度评估结果
    """
    try:
        # 1. 提取JSON片段（兼容GPT返回的多余文本，仅保留最外层{}包裹的内容）
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}") + 1
        if start_idx == -1 or end_idx == 0:
            raise ValueError("未找到有效的JSON格式内容（无{}包裹的结构）")

        json_str = response_text[start_idx:end_idx]
        # 2. 轻量清洗JSON字符串（仅修复常见格式问题，避免过度替换破坏合法内容）
        json_str = (
            json_str
            .replace("\\n", "")  # 移除换行符
            .replace("\\t", "")  # 移除制表符
            .replace("'", "\"")  # 单引号替换为双引号
            .strip()
        )

        # 3. 解析JSON
        result = json.loads(json_str)

        # 4. 校验核心字段（严格对齐instruction.txt的JSON格式要求）
        required_fields = ["score", "reasoning", "evidence_frame"]
        missing_fields = [f for f in required_fields if f not in result]
        if missing_fields:
            raise ValueError(f"维度{dimension}响应缺少关键字段：{', '.join(missing_fields)}")

        # 5. 校验evidence_frame格式（需为列表，且每个元素包含idx和observation）
        if not isinstance(result["evidence_frame"], list):
            raise ValueError(f"维度{dimension}的evidence_frame必须为列表类型，当前类型：{type(result['evidence_frame'])}")
        for idx, frame_item in enumerate(result["evidence_frame"]):
            if not isinstance(frame_item, dict) or "idx" not in frame_item or "observation" not in frame_item:
                raise ValueError(f"维度{dimension}的evidence_frame第{idx}项格式错误，需包含idx和observation字段")

        # 6. 校验score格式（需为1-5的字符串/数字）
        score = str(result["score"]).strip()
        if score not in ["1", "2", "3", "4", "5"]:
            raise ValueError(f"维度{dimension}的score值不合法（需为1-5），当前值：{result['score']}")
        result["score"] = score  # 统一转为字符串格式

        logger.debug(f"维度{dimension}响应解析成功：score={result['score']}")

        # 7. 构造返回结果（保留原始响应便于排查）
        return {
            "dimension": dimension,
            "score": result["score"],  # 严格对齐1-5字符串格式
            "reasoning": result["reasoning"].strip(),  # 评分推理过程
            "evidence_frame": result["evidence_frame"],  # 关键帧证据
            "raw_response": response_text
        }

    except json.JSONDecodeError as e:
        raise Exception(f"维度{dimension}JSON解析失败：{str(e)}，原始JSON片段：{json_str[:500]}")
    except ValueError as e:
        raise Exception(f"维度{dimension}响应校验失败：{str(e)}")
    except Exception as e:
        raise Exception(f"维度{dimension}响应解析失败：{str(e)}，原始响应：{response_text[:500]}")


def process_single_sample(sample_id: str, evaluation_spec: dict, model_name: str, prompt_template: str) -> dict:
    """
    第三步：处理单个样本，逐个维度调用GPT评估
    核心修改：对齐instruction.txt的JSON返回格式，修正占位符替换逻辑
    :param sample_id: 样本id
    :param evaluation_spec: 该样本的evaluation_specification
    :param model_name: 模型名（对应视频目录）
    :param prompt_template: 维度评估模板
    :return: 该样本的完整评估结果
    """
    try:
        logger.info(f"开始处理样本：id={sample_id} | model={model_name}")

        # 1. 获取视频关键帧Base64列表（核心调用，已适配每秒2帧）
        key_frames_base64 = get_video_key_frames_base64(
            model_name=model_name,
            sample_id=sample_id
        )

        # 2. 遍历所有维度（evaluation_spec的key即为维度名）
        dimension_results = {}
        target_dimensions = ["pg_specification", "scc_specification", "sc_specification", "lpf_specification"]
        for dimension in target_dimensions:
            # 跳过空维度内容
            dimension_content = evaluation_spec.get(dimension, "")
            if not dimension_content:
                logger.warning(f"样本{sample_id}维度{dimension}内容为空，跳过")
                # 构造空维度的默认结果（严格对齐返回格式）
                dimension_results[dimension] = {
                    "dimension": dimension,
                    "score": "0",  # 0表示未评估，区分1-5的有效评分
                    "reasoning": "维度内容为空，未执行评估",
                    "evidence_frame": [],  # 空列表符合格式要求
                    "raw_response": ""
                }
                continue

            logger.info(f"处理样本{sample_id}维度：{dimension}")

            # 2.1 构造单个维度的prompt（核心修复：按语言区分占位符）
            complete_prompt = prompt_template
            content = str(dimension_content)
            # print(content)
            if LANGUAGE == 'cn':
                # 中文模板占位符：{dimension} {dimension_content} {video_path}
                complete_prompt = complete_prompt.replace("{dimension}", evaluation_mapping[dimension]) \
                    .replace("{dimension_content}", content)
            else:
                # 英文模板占位符：{}（视频序列） {dimension} {dimension_content}
                # 英文模板第一行是 The video sequence is {}，先替换视频序列占位符
                complete_prompt = complete_prompt.replace("{dimension}", evaluation_mapping[dimension]) \
                    .replace("{dimension_content}", content)

            # 清理多余换行和空格，保证prompt格式整洁
            complete_prompt = "\n".join([line.strip() for line in complete_prompt.splitlines() if line.strip()])

            # 2.2 调用GPT接口（传入动态采样的帧列表）
            response_text = chat_gpt(complete_prompt, dimension, key_frames_base64)
            if not response_text:
                raise Exception(f"维度{dimension}GPT返回空响应")

            # 2.3 解析维度响应（严格对齐instruction.txt格式）
            dim_result = parse_gpt_response_for_dimension(response_text, dimension)
            dimension_results[dimension] = dim_result

        # 3. 构造样本完整结果（严格保留instruction.txt要求的字段）
        sample_result = {
            "id": sample_id,
            "model_name": model_name,
            "pg_specification": dimension_results.get("pg_specification", {
                "dimension": "pg_specification",
                "score": "0",
                "reasoning": "维度处理失败",
                "evidence_frame": [],
                "raw_response": ""
            }),
            "scc_specification": dimension_results.get("scc_specification", {
                "dimension": "scc_specification",
                "score": "0",
                "reasoning": "维度处理失败",
                "evidence_frame": [],
                "raw_response": ""
            }),
            "sc_specification": dimension_results.get("sc_specification", {
                "dimension": "sc_specification",
                "score": "0",
                "reasoning": "维度处理失败",
                "evidence_frame": [],
                "raw_response": ""
            }),
            "lpf_specification": dimension_results.get("lpf_specification", {
                "dimension": "lpf_specification",
                "score": "0",
                "reasoning": "维度处理失败",
                "evidence_frame": [],
                "raw_response": ""
            }),
        }
        logger.info(
            f"样本{sample_id}处理完成：共{len([v for v in dimension_results.values() if v['score'] != '0'])}个有效维度")
        return sample_result

    except Exception as e:
        logger.error(f"样本{sample_id}处理失败：{str(e)}", exc_info=True)
        raise


def save_evaluation_result(result: dict, model_name: str):
    """
    第五步：保存结果到{model_name}-evaluation.jsonl文件（追加写入）
    确保保存的JSON严格包含instruction.txt要求的score/reasoning/evidence_frame字段
    :param result: 单个样本的评估结果
    :param model_name: 模型名
    """
    output_filename = f"{model_name}-evaluation.jsonl"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    # 线程安全的追加写入
    with file_append_lock:
        try:
            with open(output_path, "a", encoding="utf-8") as f:
                # 确保JSON序列化时保留中文字符，缩进优化可读性
                f.write(json.dumps(result, ensure_ascii=False))
                f.write("\n")
            logger.debug(f"样本{result['id']}结果已追加到：{output_path}")
        except Exception as e:
            raise Exception(f"保存样本{result['id']}结果失败：{str(e)}")

# 新增：单个模型的失败样本重试函数
def retry_model_failed_samples(model_name: str, failed_sample_ids: list, jsonl_data: dict, prompt_template: str) -> tuple:
    """
    对单个模型的失败样本进行重试（最多10次）
    :param model_name: 模型名
    :param failed_sample_ids: 该模型的失败样本ID列表
    :param jsonl_data: 全局样本spec数据
    :param prompt_template: 评估模板
    :return: (retry_success_num, final_failed_ids)
    """
    retry_success = 0
    final_failed_ids = []
    current_retry_times = 0

    # 保留当前失败样本列表，用于循环重试
    current_failed_ids = failed_sample_ids.copy()

    while current_failed_ids and current_retry_times < MODEL_ERROR_RETRY_TIMES:
        current_retry_times += 1
        logger.info(f"\n模型{model_name} 第{current_retry_times}/{MODEL_ERROR_RETRY_TIMES}次重试失败样本")
        logger.info(f"本次重试样本数：{len(current_failed_ids)}")

        batch_success = 0
        batch_failed = []

        # 并发重试当前批次的失败样本
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_id = {
                executor.submit(process_single_sample, s_id, jsonl_data[s_id], model_name, prompt_template): s_id
                for s_id in current_failed_ids
            }

            for future in as_completed(future_to_id):
                sample_id = future_to_id[future]
                try:
                    sample_result = future.result()
                    save_evaluation_result(sample_result, model_name)
                    batch_success += 1
                    logger.info(f"模型{model_name} - 样本{sample_id} 第{current_retry_times}次重试成功")
                except Exception as e:
                    batch_failed.append(sample_id)
                    logger.error(f"模型{model_name} - 样本{sample_id} 第{current_retry_times}次重试失败：{str(e)}")

        # 更新统计
        retry_success += batch_success
        current_failed_ids = batch_failed
        logger.info(f"模型{model_name} 第{current_retry_times}次重试完成：成功{batch_success}条，失败{len(batch_failed)}条")

    # 最终失败的样本ID
    final_failed_ids = current_failed_ids
    return retry_success, final_failed_ids

if __name__ == "__main__":
    try:
        logger.info("=" * 60)
        global_start_time = datetime.now()
        logger.info(f"程序启动，开始时间: {global_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"并发配置：最大工作线程数 {MAX_WORKERS}，本次最大处理量 {CONTROL_NUM}")
        logger.info(f"待处理模型列表：{TARGET_MODEL_NAME}")
        logger.info(f"单个模型失败样本最大重试次数：{MODEL_ERROR_RETRY_TIMES}")
        logger.info(f"视频帧采样规则：由video_processor.py实现「每秒2帧」动态采样，缓存目录命名兼容原有规则")

        # 1. 初始化：读取模板、加载所有JSONL数据
        prompt_template = get_prompt_template_for_dimension()
        jsonl_data = load_all_jsonl_files()
        if not jsonl_data:
            logger.warning("未读取到任何JSONL样本数据，程序退出")
            exit(0)

        # 2. 筛选待处理样本（限制CONTROL_NUM）
        target_ids = list(jsonl_data.keys())[:CONTROL_NUM]
        if ENABLED_SHUFFLE:
            random.shuffle(target_ids)  # 可选：打乱样本顺序
        logger.info(f"本次计划处理样本数：{len(target_ids)}（总计：{len(jsonl_data)}）")

        # 3. 遍历每个模型，批量处理（核心修改：每个model处理完立即重试自身失败样本）
        global_success = 0
        global_failed = 0
        # 记录所有模型最终失败的样本
        all_final_failed = []

        for model_name in TARGET_MODEL_NAME:
            # 日志修改：为当前模型添加日志文件处理器
            add_model_log_handler(model_name)

            logger.info(f"\n{'=' * 40} 开始处理模型：{model_name} {'=' * 40}")
            model_start_time = datetime.now()
            processed_success = 0
            processed_failed = 0
            # 单个模型的失败样本列表
            model_failed_ids = []

            try:
                # ===================== 核心修改：过滤已处理的样本ID =====================
                # 读取该模型已处理的样本ID
                processed_ids = get_processed_sample_ids(model_name)
                # 过滤出未处理的样本ID
                unprocessed_ids = [sid for sid in target_ids if sid not in processed_ids]

                # 日志输出过滤结果
                logger.info(f"模型{model_name}样本过滤结果：")
                logger.info(f"  计划处理总数：{len(target_ids)}")
                logger.info(f"  已处理跳过数：{len(target_ids) - len(unprocessed_ids)}")
                logger.info(f"  本次实际处理数：{len(unprocessed_ids)}")

                # 无未处理样本则跳过该模型
                if not unprocessed_ids:
                    logger.info(f"模型{model_name}无未处理样本，跳过处理")
                    remove_model_log_handler(model_name)
                    continue
                # =====================================================================

                # 3.1 并发处理当前模型的**未处理**样本
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    # 提交任务：仅处理未处理的样本
                    future_to_id = {
                        executor.submit(process_single_sample, sample_id, jsonl_data[sample_id], model_name,
                                        prompt_template): sample_id
                        for sample_id in unprocessed_ids  # 核心修改：使用unprocessed_ids而非target_ids
                    }

                    # 处理完成的任务
                    for future in as_completed(future_to_id):
                        sample_id = future_to_id[future]
                        try:
                            # 获取样本处理结果
                            sample_result = future.result()
                            # 保存结果到当前model的jsonl文件
                            save_evaluation_result(sample_result, model_name)
                            processed_success += 1
                        except Exception as e:
                            logger.error(f"模型{model_name} - 样本{sample_id}任务执行失败：{str(e)}")
                            processed_failed += 1
                            model_failed_ids.append(sample_id)

                # 3.2 输出当前模型初始处理统计
                logger.info(f"\n模型{model_name}初始处理完成：")
                logger.info(f"  初始成功：{processed_success} 条")
                logger.info(f"  初始失败：{processed_failed} 条")

                # 3.3 核心修改：立即重试当前模型的失败样本（最多10次）
                if model_failed_ids:
                    retry_success, final_failed_ids = retry_model_failed_samples(
                        model_name=model_name,
                        failed_sample_ids=model_failed_ids,
                        jsonl_data=jsonl_data,
                        prompt_template=prompt_template
                    )
                    # 更新该模型的统计
                    processed_success += retry_success
                    processed_failed = len(final_failed_ids)
                    # 记录最终失败的样本
                    for s_id in final_failed_ids:
                        all_final_failed.append({"model_name": model_name, "sample_id": s_id})
                else:
                    final_failed_ids = []
                    logger.info(f"模型{model_name}无失败样本，跳过重试")

                # 3.4 输出当前模型最终统计信息
                model_duration = (datetime.now() - model_start_time).total_seconds()
                logger.info(f"\n模型{model_name}最终处理完成：")
                logger.info(f"  总耗时：{model_duration:.2f} 秒")
                logger.info(f"  最终成功：{processed_success} 条")
                logger.info(f"  最终失败：{processed_failed} 条")
                logger.info(f"  结果文件：{os.path.join(OUTPUT_DIR, f'{model_name}-evaluation.jsonl')}")
                if final_failed_ids:
                    logger.info(f"  最终失败样本ID：{final_failed_ids}")

                # 汇总到全局统计
                global_success += processed_success
                global_failed += processed_failed

            except Exception as e:
                logger.error(f"模型{model_name}处理异常：{str(e)}")
                processed_failed = len(unprocessed_ids) if 'unprocessed_ids' in locals() else len(target_ids)
                global_failed += processed_failed
                # 记录模型级异常的失败样本
                failed_ids = unprocessed_ids if 'unprocessed_ids' in locals() else target_ids
                for sample_id in failed_ids:
                    all_final_failed.append({"model_name": model_name, "sample_id": sample_id})
            finally:
                # 日志修改：处理完当前模型后，移除其日志文件处理器（避免句柄泄露）
                remove_model_log_handler(model_name)
                continue

        # 4. 输出全局统计信息
        global_duration = (datetime.now() - global_start_time).total_seconds()
        logger.info("\n" + "=" * 60)
        logger.info(f"所有模型处理完成（含各自重试），总耗时: {global_duration:.2f} 秒")
        logger.info(f"全局统计 - 成功处理: {global_success} 条，失败处理: {global_failed} 条")
        if all_final_failed:
            logger.info(f"最终失败统计 - 共{len(all_final_failed)}条样本：")
            for failed_item in all_final_failed:
                logger.info(f"  模型{failed_item['model_name']} - 样本{failed_item['sample_id']}")
        logger.info(f"输出目录：{os.path.abspath(OUTPUT_DIR)}")
        logger.info(f"日志目录：{os.path.abspath(LOG_DIR)}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"程序执行失败：{str(e)}", exc_info=True)
        exit(1)