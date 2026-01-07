import pandas as pd
import os
import json
import time
from datetime import datetime
import random
import logging
from openai import OpenAI, RateLimitError, APIError, Timeout

# ===================== 配置区域 =====================
# 建议通过环境变量设置 API Key，避免硬编码
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-WuHGSBhmDvS18EqS47C7Bd3e1dBc4412BbB884D7E7C9D0D4")
QINIUYUN_API_KEY = os.getenv("QINIUYUN_API_KEY", "sk-c6bc1786fe3cf6ab55cdcc0e7c1006b8b62a323c3b155d459cfc655ad9b4f659")
CONTROL_NUM = 3  # 每次处理的最大数据量
INPUT_DIR = "input"  # 输入目录（需与实际路径匹配）
OUTPUT_DIR = "output"  # 输出目录
API_RETRY_TIMES = 3  # API 调用失败重试次数
API_DELAY = 2  # 每次API调用间隔（秒）
LANGUAGE = 'cn'
OPENAI_MODEL = "gpt-oss-120b"  # 使用的GPT模型（gpt-4/gpt-3.5-turbo）
DATASET_FILENAME = "dataset_cn.xlsx" if LANGUAGE == 'cn' else "dataset_en.xlsx"
MAX_TOKENS = 8192  # 适配模型的最大token限制，避免响应截断
ENABLED_SHUFFLE = True
# ====================================================

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 初始化 OpenAI 客户端
# client = OpenAI(
#     base_url='https://api.qnaigc.com/v1',
#     api_key=QINIUYUN_API_KEY
# )

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://api.apiyi.com/v1"
)

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, f"{LANGUAGE}/manual"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, f"{LANGUAGE}/prompt"), exist_ok=True)


def chat_gpt(prompt: str) -> str:
    """
    调用 ChatGPT 接口，处理重试和速率限制
    :param prompt: 输入的提示词
    :return: GPT 返回的响应文本（纯字符串）
    """
    for retry in range(API_RETRY_TIMES):
        try:
            # 发送聊天请求
            response = client.chat.completions.create(
                model="gpt-5-chat-latest",
                messages=[
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                temperature=0,
                timeout=30,
                max_tokens=MAX_TOKENS
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"未知错误: {e}，重试次数：{retry + 1}/{API_RETRY_TIMES}")
            time.sleep(API_DELAY * (retry + 1))

    raise Exception(f"API 调用失败，已重试 {API_RETRY_TIMES} 次")


def get_prompt_template() -> str:
    """
    从指定文件读取 prompt 模板
    :return: 模板字符串
    """
    template_filename = "prompt_template_cn.txt" if LANGUAGE == 'cn' else "prompt_template_en.txt"
    TEMPLATE_PATH = os.path.join(INPUT_DIR, template_filename)

    try:
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
            template = f.read()
        # 读取实验说明书示例（补充到模板中）
        example_filename = "manual_example_cn.txt" if LANGUAGE == 'cn' else "manual_example_en.txt"
        example_path = os.path.join(INPUT_DIR, example_filename)
        if os.path.exists(example_path):
            with open(example_path, "r", encoding="utf-8") as f:
                example_content = f.read()
            template = template.replace("{None}", example_content)
        logger.info(f"成功读取 prompt 模板（{LANGUAGE}）")
        return template
    except FileNotFoundError:
        raise Exception(f"模板文件不存在：{TEMPLATE_PATH}")
    except Exception as e:
        raise Exception(f"读取模板文件失败：{str(e)}")


def get_complete_prompt(prompt_template: str, subject: str, sub_subject: str, requirement: str) -> str:
    """
    拼接完整的 prompt：替换模板中的占位符
    :param prompt_template: 基础模板
    :param subject: 学科
    :param sub_subject: 子学科
    :param requirement: 演示/知识点
    :return: 完整的 prompt 字符串
    """
    if not all([subject, sub_subject, requirement]):
        raise ValueError("学科、子学科、实验名称均不能为空")

    if LANGUAGE == 'cn':
        complete_prompt = prompt_template.replace("{学科}", subject) \
            .replace("{子学科}", sub_subject) \
            .replace("{实验名称}", requirement)
    elif LANGUAGE == 'en':
        complete_prompt = prompt_template.replace("{Discipline}", subject) \
            .replace("{Subdiscipline}", sub_subject) \
            .replace("{ExperimentName}", requirement)
    else:
        raise ValueError(f"不支持的语言类型：{LANGUAGE}")

    # 清理多余的换行和空格
    complete_prompt = "\n".join([line.strip() for line in complete_prompt.splitlines() if line.strip()])
    return complete_prompt


def process_excel_file(file_path: str) -> list:
    """
    处理Excel文件，提取 subject/sub_subject/requirement 字段
    :param file_path: Excel文件路径
    :return: 包含字典的列表
    """
    requirement_list = []
    try:
        excel_file = pd.ExcelFile(file_path)
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            # 列名大小写不敏感，兼容不同命名方式
            df.columns = [col.strip().lower() for col in df.columns]
            required_cols = ["sub-subject", "requirement_name"]
            required_cols_lower = [col.lower() for col in required_cols]
            if not all(col in df.columns for col in required_cols_lower):
                logger.warning(f"Sheet {sheet_name} 缺少必要列（{required_cols}），跳过该sheet")
                continue

            # 遍历行，过滤空值和无效数据
            for idx, row in df.iterrows():
                sub_subject = str(row["sub-subject"]).strip() if pd.notna(row["sub-subject"]) else ""
                requirement = str(row["requirement_name"]).strip() if pd.notna(row["requirement_name"]) else ""

                if not (sub_subject and requirement) or sub_subject == "nan" or requirement == "nan":
                    logger.warning(f"Sheet {sheet_name} 中发现空值/无效数据，跳过该行")
                    continue

                requirement_list.append({
                    "idx": f"{idx}-{sheet_name.strip()}-{sub_subject}-{requirement}",
                    "subject": sheet_name.strip(),
                    "sub_subject": sub_subject,
                    "requirement": requirement
                })

        logger.info(f"成功处理Excel文件，共提取 {len(requirement_list)} 条有效数据")
        return requirement_list

    except FileNotFoundError:
        raise Exception(f"Excel文件不存在：{file_path}")
    except Exception as e:
        raise Exception(f"处理Excel文件失败：{e}")


def parse_gpt_response(response_text: str) -> dict:
    """
    解析GPT返回的内容，提取JSON格式数据（处理可能的格式问题）
    :param response_text: GPT返回的文本
    :return: 解析后的JSON字典
    """
    try:
        # 提取JSON片段（处理GPT返回中可能的多余文本）
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}") + 1
        if start_idx == -1 or end_idx == 0:
            raise ValueError("未找到有效的JSON格式内容")

        json_str = response_text[start_idx:end_idx]
        # 修复常见的JSON格式问题
        json_str = json_str.replace("'", "\"") \
            .replace("\\n", "") \
            .replace("\\t", "")
        json_data = json.loads(json_str)

        # 校验必要字段
        required_fields = ["generation_prompt", "evaluation_rubic", "manual"]
        for field in required_fields:
            if field not in json_data:
                raise ValueError(f"缺少必要字段：{field}")

        required_rubic_fields = ["pc_rubic", "cmp_rubic", "slr_rubic", "clr_rubic", "ri_rubic"]
        for field in required_rubic_fields:
            if field not in json_data["evaluation_rubic"]:
                raise ValueError(f"evaluation_rubic 缺少必要字段：{field}")

        logger.info("GPT响应JSON解析成功")
        return json_data
    except json.JSONDecodeError as e:
        raise Exception(f"JSON解析失败：{str(e)}，原始内容片段：{response_text[:500]}")
    except ValueError as e:
        raise Exception(f"JSON格式校验失败：{str(e)}")
    except Exception as e:
        raise Exception(f"解析GPT响应失败：{str(e)}")


def save_gpt_response(data: dict, json_data: dict):
    """
    保存GPT响应结果：
    1. markdown文件 → output/manual/
    2. JSON文件 → output/prompt/
    :param data: 原始数据（subject/sub_subject/requirement）
    :param json_data: 解析后的GPT响应JSON
    """
    # 生成安全的文件名（替换特殊字符+截断过长名称）
    def safe_filename(s: str) -> str:
        if not s:
            return "unknown"
        # 替换Windows/UNIX非法字符
        unsafe_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        for char in unsafe_chars:
            s = s.replace(char, "_")
        # 截断过长文件名（避免系统限制）
        return s[:50]  # 限制最大长度50字符

    base_filename = safe_filename(data["idx"])

    # 保存manual文件
    try:
        markdown_content = json_data["manual"]
        markdown_path = os.path.join(OUTPUT_DIR, f"{LANGUAGE}/manual", f"{base_filename}.md")
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        logger.info(f"manual文件已保存：{markdown_path}")
    except Exception as e:
        raise Exception(f"保存manual文件失败：{e}")

    # 保存JSON文件
    try:
        json_path = os.path.join(OUTPUT_DIR, f"{LANGUAGE}/prompt", f"{base_filename}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
        logger.info(f"prompt文件已保存：{json_path}")
    except Exception as e:
        raise Exception(f"保存prompt文件失败：{e}")


if __name__ == "__main__":
    try:
        logger.info("=" * 60)
        start_time = datetime.now()
        logger.info(f"程序启动，开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # 1. 读取模板和Excel数据
        prompt_template = get_prompt_template()
        excel_path = os.path.join(INPUT_DIR, DATASET_FILENAME)
        requirement_list = process_excel_file(excel_path)

        # 2. 限制处理数量
        process_limit = min(len(requirement_list), CONTROL_NUM)
        logger.info(f"本次处理数量：{process_limit}（总计有效数据：{len(requirement_list)}）")

        # 3. 遍历处理每条数据
        processed_success = 0
        processed_failed = 0

        for idx in range(process_limit):
            if ENABLED_SHUFFLE:
                index = random.randint(0, len(requirement_list)-1)  # 可能重复抽中同一个，测试功能没影响
            else:
                index = idx

            try:
                data = requirement_list[index]
                subject = data["subject"]
                sub_subject = data["sub_subject"]
                requirement = data["requirement"]

                logger.info(f"\n处理第 {idx + 1}/{process_limit} 条：")
                logger.info(f"学科：{subject} | 子学科：{sub_subject} | 演示目标：{requirement}")

                # 拼接完整Prompt
                chat_prompt = get_complete_prompt(prompt_template, subject, sub_subject, requirement)

                # 调用GPT接口
                response_text = chat_gpt(chat_prompt)
                if not response_text:
                    raise ValueError("GPT返回空响应")

                # 解析GPT响应
                json_data = parse_gpt_response(response_text)

                # 保存结果
                save_gpt_response(data, json_data)

                # 计数+延迟（最后一条不延迟）
                processed_success += 1
                if idx < process_limit - 1:
                    time.sleep(API_DELAY)

            except Exception as e:
                logger.error(f"处理第 {idx + 1} 条数据失败：{str(e)}", exc_info=True)
                processed_failed += 1
                continue

        # 输出统计信息
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info("=" * 60)
        logger.info(f"程序结束，结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"总耗时: {duration:.2f} 秒")
        logger.info(f"成功处理: {processed_success} 条")
        logger.info(f"失败处理: {processed_failed} 条")
        logger.info("=" * 60)

        # TODO Json文件合并和追加拼接

    except Exception as e:
        logger.error(f"程序执行失败：{str(e)}", exc_info=True)
        exit(1)