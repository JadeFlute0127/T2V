import os
import json
import csv
import statistics
from typing import Dict, List, Tuple, Any
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ===================== 配置区域 =====================
# 原始样本数据目录（input/dataset下的jsonl文件）
RAW_DATA_DIR = "input/dataset"
# 评测结果目录（eval/output下的model_name.jsonl文件）
EVAL_RESULT_DIR = "output"
# 输出结果目录（自动创建）
OUTPUT_DIR = "analysis_results"
# 需要分析的评测维度
EVAL_DIMENSIONS = [
    "pg_specification",
    "scc_specification",
    "sc_specification",
    "lpf_specification"
]
# 维度名称映射（用于更友好的展示）
DIMENSION_NAME_MAP = {
    "pg_specification": "Prompt Grounding",
    "scc_specification": "Scientific & Causal Correctness",
    "sc_specification": "Spatiotemporal Consistency",
    "lpf_specification": "Low-level Perceptual Fidelity"
}
# 可视化配置
PLOT_DPI = 300  # 图片清晰度（越高越清晰）
PLOT_FIGSIZE = (16, 10)  # 基础图表尺寸
# 解决matplotlib中文显示问题（根据系统适配）
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]  # Windows/Linux
plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示
# ====================================================

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def load_raw_data() -> Dict[str, Dict[str, str]]:
    """
    读取原始样本数据，构建id到domain、subject的映射
    返回格式：{sample_id: {"domain": xxx, "subject": xxx}}
    """
    id_metadata_map = {}
    id_list = []
    logger.info(f"开始读取原始样本数据：{RAW_DATA_DIR}")

    if not os.path.exists(RAW_DATA_DIR):
        logger.error(f"原始样本数据目录不存在：{RAW_DATA_DIR}")
        return id_metadata_map

    # 遍历所有jsonl文件
    for filename in os.listdir(RAW_DATA_DIR):
        if not filename.endswith(".jsonl") or filename == "sample.jsonl":
            continue

        file_path = os.path.join(RAW_DATA_DIR, filename)
        logger.debug(f"读取原始文件：{file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    sample_data = json.loads(line)
                    sample_id = sample_data.get("id")
                    if not sample_id:
                        logger.warning(f"{file_path}第{line_num}行：缺少id字段，跳过")
                        continue

                    # 提取domain和subject（兼容不同字段名）
                    domain = sample_data.get("domain", sample_data.get("Domain", "未知"))
                    subject = sample_data.get("subject", sample_data.get("Subject", "未知"))

                    if sample_id in id_list:
                        logger.warning(f"{domain},{subject},{sample_id}重复")
                        exit()

                    id_metadata_map[sample_id] = {
                        "domain": domain.strip(),
                        "subject": subject.strip()
                    }
                    id_list.append(sample_id)
                except json.JSONDecodeError as e:
                    logger.warning(f"{file_path}第{line_num}行：JSON解析失败，跳过 | 错误：{str(e)}")
                except Exception as e:
                    logger.warning(f"{file_path}第{line_num}行：处理失败，跳过 | 错误：{str(e)}")

    logger.info(f"原始样本数据读取完成，共加载{len(id_metadata_map)}个样本的元数据")
    return id_metadata_map

def load_eval_results(id_metadata_map: Dict[str, Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    读取评测结果，关联原始元数据，生成聚合分析的原始数据列表
    返回格式：[{
        "sample_id": xxx,
        "model_name": xxx,
        "domain": xxx,
        "subject": xxx,
        "pg_specification": 1-5,
        "scc_specification": 1-5,
        "sc_specification": 1-5,
        "lpf_specification": 1-5
    }, ...]
    """
    aggregated_raw_data = []
    logger.info(f"开始读取评测结果数据：{EVAL_RESULT_DIR}")

    if not os.path.exists(EVAL_RESULT_DIR):
        logger.error(f"评测结果目录不存在：{EVAL_RESULT_DIR}")
        return aggregated_raw_data

    # 遍历所有model的评测结果文件
    for filename in os.listdir(EVAL_RESULT_DIR):
        if not filename.endswith(".jsonl"):
            continue

        model_name = filename.replace(".jsonl", "")
        file_path = os.path.join(EVAL_RESULT_DIR, filename)
        logger.debug(f"读取评测结果文件：{file_path}（模型：{model_name}）")

        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    eval_result = json.loads(line)
                    sample_id = eval_result.get("id")
                    if not sample_id:
                        logger.warning(f"{file_path}第{line_num}行：缺少id字段，跳过")
                        continue

                    # 获取原始元数据
                    metadata = id_metadata_map.get(sample_id, {})
                    domain = metadata.get("domain", "未知")
                    subject = metadata.get("subject", "未知")

                    # 解析各维度得分（校验得分有效性）
                    dimension_scores = {}
                    for dim in EVAL_DIMENSIONS:
                        dim_data = eval_result.get(dim, {})
                        score_str = dim_data.get("score", "0")
                        try:
                            score = int(score_str)
                            if score < 1 or score > 5:
                                logger.warning(f"{file_path}第{line_num}行（样本{sample_id}）：{dim}得分{score}无效（需1-5），标记为0")
                                score = 0
                        except (ValueError, TypeError):
                            logger.warning(f"{file_path}第{line_num}行（样本{sample_id}）：{dim}得分{score_str}不是数字，标记为0")
                            score = 0
                        dimension_scores[dim] = score

                    # 构建聚合原始数据
                    aggregated_raw_data.append({
                        "sample_id": sample_id,
                        "model_name": model_name,
                        "domain": domain,
                        "subject": subject,
                        **dimension_scores
                    })
                except json.JSONDecodeError as e:
                    logger.warning(f"{file_path}第{line_num}行：JSON解析失败，跳过 | 错误：{str(e)}")
                except Exception as e:
                    logger.warning(f"{file_path}第{line_num}行：处理失败，跳过 | 错误：{str(e)}")

    logger.info(f"评测结果读取完成，共加载{len(aggregated_raw_data)}个有效评测样本")
    return aggregated_raw_data

def calculate_statistics(scores: List[int]) -> Dict[str, Any]:
    """
    计算得分的统计指标（过滤掉0分无效值）
    返回：包含平均分、标准差、计数、最高分、最低分的字典
    """
    # 过滤无效得分（0分）
    valid_scores = [s for s in scores if s > 0]
    if not valid_scores:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0,
            "max": 0
        }

    return {
        "count": len(valid_scores),
        "mean": round(statistics.mean(valid_scores), 2),
        "std": round(statistics.stdev(valid_scores) if len(valid_scores) > 1 else 0.0, 2),
        "min": min(valid_scores),
        "max": max(valid_scores)
    }

def aggregate_analysis(aggregated_raw_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    执行多维度聚合分析（新增模型×维度×domain三维数据）
    """
    logger.info("开始执行聚合分析...")
    analysis_results = {}

    # 1. 整体维度得分统计
    logger.info("  - 计算整体维度得分统计")
    overall_stats = {}
    for dim in EVAL_DIMENSIONS:
        all_scores = [item[dim] for item in aggregated_raw_data]
        overall_stats[dim] = calculate_statistics(all_scores)
    analysis_results["overall"] = overall_stats

    # 2. 按模型分组统计
    logger.info("  - 按模型分组统计")
    model_groups = {}
    for item in aggregated_raw_data:
        model = item["model_name"]
        if model not in model_groups:
            model_groups[model] = []
        model_groups[model].append(item)

    model_stats = {}
    for model, items in model_groups.items():
        model_dim_stats = {}
        for dim in EVAL_DIMENSIONS:
            scores = [item[dim] for item in items]
            model_dim_stats[dim] = calculate_statistics(scores)
        model_stats[model] = model_dim_stats
    analysis_results["by_model"] = model_stats

    # 3. 按领域（domain）分组统计
    logger.info("  - 按领域分组统计")
    domain_groups = {}
    for item in aggregated_raw_data:
        domain = item["domain"]
        if domain not in domain_groups:
            domain_groups[domain] = []
        domain_groups[domain].append(item)

    domain_stats = {}
    for domain, items in domain_groups.items():
        domain_dim_stats = {}
        for dim in EVAL_DIMENSIONS:
            scores = [item[dim] for item in items]
            domain_dim_stats[dim] = calculate_statistics(scores)
        domain_stats[domain] = domain_dim_stats
    analysis_results["by_domain"] = domain_stats

    # 4. 按学科（subject）分组统计
    logger.info("  - 按学科分组统计")
    subject_groups = {}
    for item in aggregated_raw_data:
        subject = item["subject"]
        if subject not in subject_groups:
            subject_groups[subject] = []
        subject_groups[subject].append(item)

    subject_stats = {}
    for subject, items in subject_groups.items():
        subject_dim_stats = {}
        for dim in EVAL_DIMENSIONS:
            scores = [item[dim] for item in items]
            subject_dim_stats[dim] = calculate_statistics(scores)
        subject_stats[subject] = subject_dim_stats
    analysis_results["by_subject"] = subject_stats

    # 5. 原始明细数据（用于CSV输出）
    analysis_results["raw_detail"] = aggregated_raw_data

    # 6. 新增：模型×维度×domain三维聚合数据（绘图专用）
    logger.info("  - 计算模型×维度×domain三维聚合数据")
    model_dim_domain_stats = {}
    # 按模型分组
    for model, model_items in model_groups.items():
        model_dim_domain_stats[model] = {}
        # 按维度分组
        for dim in EVAL_DIMENSIONS:
            model_dim_domain_stats[model][dim] = {}
            # 按domain统计有效得分的平均分
            domain_score_map = {}
            for item in model_items:
                domain = item["domain"]
                score = item[dim]
                if score == 0:  # 过滤无效分
                    continue
                if domain not in domain_score_map:
                    domain_score_map[domain] = []
                domain_score_map[domain].append(score)
            # 计算每个domain的平均分
            for domain, scores in domain_score_map.items():
                model_dim_domain_stats[model][dim][domain] = round(statistics.mean(scores), 2)
    analysis_results["model_dim_domain"] = model_dim_domain_stats

    logger.info("聚合分析完成！")
    return analysis_results

def plot_3d_score_distribution(analysis_results: Dict[str, Any]):
    """
    绘制模型×维度×domain三维分数分布图（3类核心图表）
    """
    logger.info("开始生成三维分数分布图...")
    model_dim_domain = analysis_results.get("model_dim_domain", {})
    if not model_dim_domain:
        logger.warning("无模型×维度×domain聚合数据，跳过绘图")
        return

    # 提取基础数据
    models = list(model_dim_domain.keys())
    dimensions = EVAL_DIMENSIONS
    # 收集所有唯一的domain（去重并排序）
    all_domains = set()
    for model in models:
        for dim in dimensions:
            all_domains.update(model_dim_domain[model][dim].keys())
    all_domains = sorted(list(all_domains))
    if not all_domains:
        logger.warning("无有效domain数据，跳过绘图")
        return

    # 定义模型颜色映射（区分不同模型）
    colors = sns.color_palette("husl", n_colors=len(models))
    model_color = dict(zip(models, colors))

    # ========== 图表1：分组柱状图（维度×模型×domain） ==========
    # 每个维度单独一个子图，展示不同模型在各domain的得分
    logger.info("  - 生成分组柱状图（维度×模型×domain）")
    fig, axes = plt.subplots(len(dimensions), 1, figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    if len(dimensions) == 1:
        axes = [axes]  # 兼容单维度场景

    for idx, dim in enumerate(dimensions):
        ax = axes[idx]
        # 每个domain为一组，组内展示不同模型的柱子
        x = np.arange(len(all_domains))
        bar_width = 0.8 / len(models)  # 自适应柱子宽度

        # 绘制每个模型的柱子
        for i, model in enumerate(models):
            # 获取该模型该维度下各domain的得分（无数据则为0）
            scores = [model_dim_domain[model][dim].get(domain, 0) for domain in all_domains]
            # 计算柱子位置（居中对齐）
            bar_pos = x + (i - len(models)/2 + 0.5) * bar_width
            ax.bar(
                bar_pos, scores, bar_width,
                label=model, color=model_color[model], alpha=0.8
            )

        # 图表样式优化
        ax.set_title(f"{DIMENSION_NAME_MAP.get(dim, dim)} 得分对比（按Domain）", fontsize=12, fontweight="bold")
        ax.set_ylabel("平均分", fontsize=10)
        ax.set_xlabel("Domain", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(all_domains, rotation=45, ha="right")
        ax.set_ylim(0, 5.5)  # 得分范围固定为0-5（贴合评测规则）
        ax.grid(axis="y", alpha=0.3)
        ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    barplot_path = os.path.join(OUTPUT_DIR, "model_dim_domain_barplot.png")
    plt.savefig(barplot_path, bbox_inches="tight")
    logger.info(f"    - 分组柱状图已保存：{barplot_path}")
    plt.close()

    # ========== 图表3：雷达图（Domain×模型×维度） ==========
    # 每个domain单独生成雷达图，展示不同模型的维度得分分布
    logger.info("  - 生成Domain维度雷达图")
    # 雷达图角度配置（闭合图形）
    angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
    angles += angles[:1]  # 闭合雷达图

    for domain in all_domains:
        fig, ax = plt.subplots(figsize=(10, 10), dpi=PLOT_DPI, subplot_kw=dict(polar=True))
        # 绘制每个模型的雷达图
        for model in models:
            # 获取该模型在该domain下各维度的得分（闭合）
            scores = [model_dim_domain[model][dim].get(domain, 0) for dim in dimensions]
            scores += scores[:1]  # 闭合图形
            # 绘制雷达图
            ax.plot(angles, scores, 'o-', linewidth=2, label=model, color=model_color[model])
            ax.fill(angles, scores, alpha=0.1, color=model_color[model])

        # 雷达图样式优化
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([DIMENSION_NAME_MAP.get(dim, dim) for dim in dimensions], fontsize=10)
        ax.set_ylim(0, 5)
        ax.set_title(f"Domain: {domain} - 各模型维度得分雷达图", fontsize=12, fontweight="bold", pad=20)
        ax.grid(True)
        ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.0))

        # 保存雷达图（处理特殊字符）
        safe_domain_name = domain.replace("/", "_").replace("\\", "_")
        radar_path = os.path.join(OUTPUT_DIR, f"domain_{safe_domain_name}_radarplot.png")
        plt.tight_layout()
        plt.savefig(radar_path)
        logger.info(f"    - {domain}雷达图已保存：{radar_path}")
        plt.close()

    logger.info("三维分数分布图生成完成！")

def save_analysis_results(analysis_results: Dict[str, Any]):
    """
    保存分析结果到文件（JSON+CSV+TXT，保留原有逻辑）
    """
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"开始保存分析结果到：{OUTPUT_DIR}")

    # 1. 保存聚合统计结果（JSON格式）
    json_output_path = os.path.join(OUTPUT_DIR, "aggregation_stats.json")
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=2)
    logger.info(f"  - 聚合统计结果已保存：{json_output_path}")

    # 2. 保存原始明细数据（CSV格式，方便Excel查看）
    csv_output_path = os.path.join(OUTPUT_DIR, "raw_detail.csv")
    if analysis_results.get("raw_detail"):
        raw_detail = analysis_results["raw_detail"]
        headers = ["sample_id", "model_name", "domain", "subject"] + EVAL_DIMENSIONS
        with open(csv_output_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(raw_detail)
        logger.info(f"  - 原始明细数据已保存：{csv_output_path}")

    # 3. 生成友好的统计报告（TXT格式）
    report_output_path = os.path.join(OUTPUT_DIR, "analysis_report.txt")
    with open(report_output_path, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("视频评测结果聚合分析报告\n")
        f.write("="*80 + "\n\n")

        # 整体统计
        f.write("1. 整体维度得分统计\n")
        f.write("-"*50 + "\n")
        for dim, stats in analysis_results["overall"].items():
            f.write(f"{DIMENSION_NAME_MAP.get(dim, dim)}:\n")
            f.write(f"  有效样本数：{stats['count']}\n")
            f.write(f"  平均分：{stats['mean']}\n")
            f.write(f"  标准差：{stats['std']}\n")
            f.write(f"  最高分：{stats['max']}\n")
            f.write(f"  最低分：{stats['min']}\n\n")

        # 按模型统计
        f.write("2. 按模型分组统计\n")
        f.write("-"*50 + "\n")
        for model, dim_stats in analysis_results["by_model"].items():
            f.write(f"模型：{model}\n")
            for dim, stats in dim_stats.items():
                f.write(f"  {DIMENSION_NAME_MAP.get(dim, dim)}：平均分={stats['mean']}（有效样本数={stats['count']}）\n")
            f.write("\n")

        # 按领域统计
        f.write("3. 按领域分组统计\n")
        f.write("-"*50 + "\n")
        for domain, dim_stats in analysis_results["by_domain"].items():
            f.write(f"领域：{domain}\n")
            for dim, stats in dim_stats.items():
                f.write(f"  {DIMENSION_NAME_MAP.get(dim, dim)}：平均分={stats['mean']}（有效样本数={stats['count']}）\n")
            f.write("\n")

        # 按学科统计
        f.write("4. 按学科分组统计\n")
        f.write("-"*50 + "\n")
        for subject, dim_stats in analysis_results["by_subject"].items():
            f.write(f"学科：{subject}\n")
            for dim, stats in dim_stats.items():
                f.write(f"  {DIMENSION_NAME_MAP.get(dim, dim)}：平均分={stats['mean']}（有效样本数={stats['count']}）\n")
            f.write("\n")

    logger.info(f"  - 分析报告已保存：{report_output_path}")
    logger.info("所有结果保存完成！")

def main():
    """主函数（保留原有逻辑，新增绘图调用）"""
    try:
        logger.info("="*80)
        logger.info("视频评测结果聚合分析脚本启动")
        logger.info("="*80)

        # 步骤1：读取原始样本数据
        id_metadata_map = load_raw_data()
        if not id_metadata_map:
            logger.error("未读取到任何原始样本数据，脚本退出")
            return

        # 步骤2：读取评测结果并关联元数据
        aggregated_raw_data = load_eval_results(id_metadata_map)
        if not aggregated_raw_data:
            logger.error("未读取到任何有效评测结果，脚本退出")
            return

        # 步骤3：执行聚合分析（含三维数据）
        analysis_results = aggregate_analysis(aggregated_raw_data)

        # 步骤4：保存原有3个分析文件
        save_analysis_results(analysis_results)

        # 步骤5：新增：绘制三维分数分布图
        plot_3d_score_distribution(analysis_results)

        logger.info("="*80)
        logger.info("脚本执行完成！分析结果已保存至：%s", OUTPUT_DIR)
        logger.info("="*80)

    except Exception as e:
        logger.error(f"脚本执行失败：{str(e)}", exc_info=True)
        exit(1)

if __name__ == "__main__":
    main()