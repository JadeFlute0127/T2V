import json
import re


def process_json_string(raw_json_str: str) -> dict:
    """
    处理包含多层转义、非法转义、格式瑕疵的JSON字符串，返回解析后的字典
    适配场景：GPT返回的JSON字符串（含\\n、\\\\n、多余引号、全角标点等问题）

    Args:
        raw_json_str: 原始的JSON字符串（可能包含格式问题）

    Returns:
        dict: 解析后的合法JSON字典

    Raises:
        ValueError: JSON字符串为空/关键字段缺失
        json.JSONDecodeError: 修复后仍无法解析（含详细错误位置）
    """
    # 步骤1：基础清理（处理首尾多余引号/空白）
    cleaned_str = raw_json_str.strip()
    # 移除首尾可能的外层引号（比如输入字符串被额外包裹了一层""）
    if cleaned_str.startswith('"') and cleaned_str.endswith('"'):
        cleaned_str = cleaned_str[1:-1]
    # 移除首尾空白字符
    cleaned_str = cleaned_str.strip()

    if not cleaned_str:
        raise ValueError("输入的JSON字符串为空")

    # 步骤2：修复转义符问题（核心）
    fix_str = cleaned_str
    # 2.1 修复多层转义（\\\\n → \\n，GPT常返回的多余反斜杠）
    fix_str = fix_str.replace("\\\\n", "\\n").replace("\\\\t", "\\t")
    # 2.2 修复非法转义序列（\1、\2、\3 → \n1、\n2、\n3，常见笔误）
    fix_str = re.sub(r'\\(\d)', r'\\n\1', fix_str)
    # 2.3 修复全角标点（替换为半角，避免解析错误）
    fix_str = fix_str.replace("：", ":").replace("，", ",").replace("”", "\"").replace("“", "\"")
    fix_str = fix_str.replace("；", ";").replace("。", ".").replace("（", "(").replace("）", ")")

    # 步骤3：修复JSON格式瑕疵
    # 3.1 移除最后一个元素后的多余逗号（比如 "key":"value", } → "key":"value" }）
    fix_str = re.sub(r',\s*}', '}', fix_str)
    fix_str = re.sub(r',\s*]', ']', fix_str)
    # 3.2 确保属性名被双引号包裹（防止GPT返回无引号的键名）
    fix_str = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', fix_str)
    # 3.3 修复单引号（如果有）→ 双引号（排除已转义的单引号）
    fix_str = re.sub(r'(?<!\\)\'', '"', fix_str)

    # 步骤4：解析JSON并捕获详细错误
    try:
        json_data = json.loads(fix_str)
    except json.JSONDecodeError as e:
        # 抛出包含错误位置和原始修复后字符串的异常，便于调试
        error_msg = (
            f"JSON解析失败！\n"
            f"错误位置：行{e.lineno}，列{e.colno}（字符位置{e.pos}）\n"
            f"错误原因：{e.msg}\n"
            f"修复后的字符串前500字符：\n{fix_str[:500]}..."
        )
        raise json.JSONDecodeError(error_msg, e.doc, e.pos) from e

    # 步骤5：验证关键字段（确保解析结果符合业务预期）
    required_fields = ["generation_prompt", "evaluation_rubic", "manual"]
    missing_fields = [f for f in required_fields if f not in json_data]
    if missing_fields:
        raise ValueError(f"解析后的JSON缺失关键字段：{missing_fields}")

    # 额外验证evaluation_rubic的子字段（可选，根据你的业务需求调整）
    rubic_fields = ["pc_rubic", "cmp_rubic", "slr_rubic", "clr_rubic", "ri_rubic"]
    rubic_data = json_data.get("evaluation_rubic", {})
    missing_rubic = [f for f in rubic_fields if f not in rubic_data]
    if missing_rubic:
        raise ValueError(f"evaluation_rubic缺失子字段：{missing_rubic}")

    return json_data


# ===================== 测试示例（使用你提供的JSON字符串） =====================
if __name__ == "__main__":
    # 你提供的原始JSON字符串
    test_json_str = "\"{\\n\\\"generation_prompt\\\": \\\"Laboratory Preparation of Hydrogen by Downward Displacement of Air. Show a clear laboratory setup: a zinc granule sample in a conical flask connected via a glass delivery tube to an inverted gas jar over the flask mouth. A student slowly adds dilute hydrochloric acid to the zinc in the flask. Bubbles of hydrogen gas are formed vigorously, displacing air in the jar from the top down. The entire gas jar gradually fills with colorless hydrogen. After collection, a burning splint is brought near the mouth of the jar, producing a characteristic soft ‘pop’ sound. The laboratory is bright and clean, glassware realistic, liquid colorless, and labels simple and readable.\\\",\\n\\\"evaluation_rubic\\\": {\\n\\\"pc_rubic\\\": \\\"Hard Rules: (1) The scene must show zinc and dilute hydrochloric acid in a conical flask connected to a gas-collecting jar by a delivery tube. (2) The steps should occur in the correct order: acid addition → gas generation → downward displacement collection → ignition test. Principles: The appearance and behavior of materials must conform to inorganic chemistry background knowledge—zinc is metallic gray, hydrochloric acid colorless, hydrogen invisible; the displacement process should match typical laboratory practice.\\\",\\n\\\"cmp_rubic\\\": \\\"Hard Rules: The main phenomenon—formation of continuous gas bubbles, collection of a colorless gas by downward displacement, and the 'pop' sound when tested with a flame—must all be clearly presented in sequence. Principles: The brightness of gas bubbles, motion of liquid, and sound must be realistic; all visible changes must match the expected hydrogen preparation phenomenon.\\\",\\n\\\"slr_rubic\\\": \\\"Hard Rules: The video should display smooth temporal and spatial continuity between adding acid, gas generation, gas collection, and ignition testing. Principles: Temporal Logic—gas generation accelerates gradually after acid addition; Spatial Logic—bubbles rise correctly in liquid, gas jar position consistent; Spatiotemporal Consistency—each action continues smoothly without abrupt or impossible transitions.\\\",\\n\\\"clr_rubic\\\": \\\"Hard Rules: The cause (acid reacting with zinc) must directly lead to gas generation, which then leads logically to hydrogen collection and ignition testing. Principles: Object State Logic—bubbles appear because of Zn and HCl reaction; Scientific Principle Logic—the produced hydrogen reacts violently with a flame giving a small explosion; Presentation—cause and effect should be intuitive and simple.\\\",\\n\\\"ri_rubic\\\": \\\"Hard Rules: Throughout the process, all chemical and physical properties must comply with real scientific laws—no color change of hydrogen, no floating opposite to physical laws. Principles: Stability of Object Properties—glass remains rigid, acid remains liquid, gas invisible; Human Actions—motion consistent with real laboratory handling.\\\"\\n},\\n\\\"manual\\\": \\\"I. Basic Experiment Information\\\\\\\\n1. Experiment Title: Laboratory Preparation of Hydrogen by Downward Displacement of Air\\\\\\\\n2. Subject: Chemistry - Inorganic Chemistry\\\\\\\\n3. Experiment Objective: To learn the laboratory preparation method of hydrogen, understand its physical properties (such as being colorless, odorless, lighter than air), and grasp the principle of metal and acid reaction generating hydrogen.\\\\\\\\n\\\\\\\\nII. Experimental Equipment and Materials\\\\\\\\nConical flask (100 mL), single-hole rubber stopper, glass delivery tube, gas jar, trough or support ring for jar, zinc granules, dilute hydrochloric acid (1:5 HCl), dropper or funnel, wooden splint, lighter or match, safety goggles and gloves.\\\\\\\\n\\\\\\\\nIII. Experimental Procedure\\\\\\\\n1. Preparation Stage: Wear goggles and gloves. Check that all glassware are clean and dry. Place several zinc granules into a 100 mL conical flask.\\\\\\\\n2. Equipment Assembly: Fit the flask with a single-hole rubber stopper attached to a glass delivery tube. Adjust the delivery tube so that its other end leads into the mouth of an inverted gas jar resting above the flask (no water seal needed since gas is collected by downward air displacement).\\\\\\\\n3. Gas Generation: Using a dropper, slowly add an appropriate amount of dilute hydrochloric acid into the conical flask. Immediately note the release of bubbles. Oxygen or air in the setup should be allowed to escape for a few seconds to flush; then collection begins.\\\\\\\\n4. Gas Collection: Hydrogen gas generated in the flask flows through the delivery tube into the gas jar, pushing the air inside downward and out from the bottom. Continue collection until the jar is full of hydrogen. Cover the mouth of the jar with a lid or glass plate to prevent gas loss.\\\\\\\\n5. Gas Test: Bring a burning splint near the mouth of the collected jar and slightly lift the cover. A soft 'pop' sound confirms the presence of hydrogen.\\\\\\\\n6. Post-Experiment Handling: Remove the delivery tube before stopping acid addition to prevent liquid backflow. Pour waste solutions into designated waste containers, rinse and clean all equipment, and restore the workspace.\\\\\\\\n\\\\\\\\nIV. Observation of Experimental Phenomena\\\\\\\\n1. Initial State: Zinc granules appear metallic gray in colorless dilute hydrochloric acid.\\\\\\\\n2. During Reaction: After acid is added, vigorous effervescence (continuous bubble formation) is observed.\\\\\\\\n3. Gas Collection: The gas collected is colorless and invisible; the jar shows no condensation.\\\\\\\\n4. Gas Test: When a burning splint is touched to the jar mouth, a soft 'pop' sound is heard, confirming hydrogen gas.\\\\\\\\n\\\\\\\\nV. Summary of Experimental Principles\\\\\\\\nHydrogen is produced through the reaction of a metal with an acid. Zinc reacts with dilute hydrochloric acid to yield hydrogen gas and zinc chloride solution. The generated hydrogen is lighter than air, so it can be collected by downward displacement of air.\\\\\\\\nReaction Equation: Zn + 2HCl → ZnCl₂ + H₂↑\\\\\\\\nThis method demonstrates both the chemical reactivity of active metals with acids and the method for collecting low-density gases.\\\\\\\\n\\\\\\\\nVI. Experimental Precautions\\\\\\\\n1. Safety: Always wear goggles and gloves. Avoid flames near the apparatus until the hydrogen collection is complete.\\\\\\\\n2. Operation: Add acid slowly to prevent violent reaction or splashing. Ensure all joints are sealed to avoid hydrogen leakage.\\\\\\\\n3. Sequence: Always remove the stopper before ending the reaction to prevent liquid backflow.\\\\\\\\n4. Waste Disposal: Collect and neutralize waste acid properly, and clean the apparatus thoroughly.\\\\\\\\n\\\"\\n}\""

    try:
        # 调用处理函数
        result = process_json_string(test_json_str)
        print("✅ JSON处理并解析成功！")
        # 打印关键字段预览，验证结果
        print("\n📌 generation_prompt 预览：")
        print(result["generation_prompt"][:100] + "...")
        print("\n📌 manual 字段行数：", len(result["manual"].split("\\n")))
        print("\n📌 evaluation_rubic 包含的子字段：", list(result["evaluation_rubic"].keys()))
    except (json.JSONDecodeError, ValueError) as e:
        print("❌ 处理失败：", e)
