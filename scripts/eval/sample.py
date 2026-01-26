import os
import random
import json

sample_count = 2

black_list = [
    "39691e15c62cb0b9ba072532e82eb91ca1f20d38f",
    "709827b270c049c4838de2fd1205af4bf9f59446",
    "733d0061a45c022ad410ff6af7356d443484378e",
    "709827b270c049c4838de2fd1205af4bf9f59312",
    "5f92a5e92f0768c580d3e0a34d70d15bd1155652",
    "31882e2e1a77accc6aae950ee167f3f4d2222ac1",
    "fe828e5d44adfbbb38a7e4615a7becb75990548d",
    "39691e15c62cb0b9ba072532e82eb91ca1f20d42f",
    "535d9102895e2729995d0168d47125fd80b132ddb",
    #
    "39691e15c62cb0b9ba072532e82eb91ca1f20d33f",
    "39691e15c62cb0b9ba072532e82eb91ca1f20d47f",
    "39691e15c62cb0b9ba072532e82eb91ca1f20d45f",
    "c42e7145ffc814029700f7a03aedaffa9add6bc63",
    "4ce9cea21f3e28dc3109117ae2d88f6a8ffe2214",
    "709827b270c049c4838de2fd1205af4bf9f59369",
    "cae9d161809fb58150390f079d94cd4784fb6623",
    "c42e7145ffc814029700f7a03aedaffa9add6bc28",
    "c42e7145ffc814029700f7a03aedaffa9add6bc62",
    "3886f81f37f64b6426993bae13d611616af441ee",
    "2e67b5a82fd3f9d5a9abf463d1a1cb187b3c2f59",
    #
    "40077c3579c6271850e0e40d228c441102aeb1fd",
    #
    "58e7ae166afa2f10a40af454cea4caf4b03632e216",
    "23aad5a477e51536fcda926c06bdac0ed8772bdc",
    "1bb9bbfe60539d5fc587939959ea19603b5a5d1a"
]

def sample_jsonl_files():
    existing_samples = load_all_jsonl_files()
    # 定义数据集目录路径
    dataset_dir = "input/dataset"
    # 目标采样数量
    # 存储所有有效样本
    all_valid_samples = []

    # 遍历目录下的所有文件
    for filename in os.listdir(dataset_dir):
        # 仅处理jsonl文件，且排除输出文件本身
        if filename.endswith(".jsonl") and filename != "sample.jsonl":
            file_path = os.path.join(dataset_dir, filename)
            print(f"正在读取文件: {file_path}")

            # 读取文件中的每一行样本
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    # 去除首尾空白字符
                    line = line.strip()
                    # 跳过空行
                    if not line:
                        continue

                    # 验证JSON格式有效性，避免无效样本
                    try:
                        line_json_data = json.loads(line)
                        if line_json_data.get("id") in existing_samples or line_json_data.get("id") in black_list:
                            continue
                        all_valid_samples.append(line)
                    except json.JSONDecodeError as e:
                        print(f"警告：文件{filename}第{line_num}行JSON格式无效，已跳过 | 错误信息: {str(e)}")

    # 检查总样本数是否满足采样需求
    total_samples = len(all_valid_samples)
    if total_samples < sample_count:
        print(f"警告：总有效样本数({total_samples})不足{sample_count}个，将使用所有样本")
        sampled_samples = all_valid_samples
    else:
        # 随机采样（无重复）
        sampled_samples = random.sample(all_valid_samples, sample_count)

    # 定义输出文件路径
    output_path = os.path.join(dataset_dir, "sample2.jsonl")

    # 将采样结果写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in sampled_samples:
            f.write(sample + '\n')

    # 输出完成信息
    print(f"\n采样完成！")
    print(f"总有效样本数: {total_samples}")
    print(f"实际采样数: {len(sampled_samples)}")
    print(f"采样结果已保存至: {output_path}")



def load_all_jsonl_files() -> list:
    """
    第一步：读取/input/dataset目录下的所有jsonl文件，按id聚合数据
    :return: 字典 {id: evaluation_specification}
    """
    jsonl_data = []

    file_path = os.path.join("./input", "sample.jsonl")

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                jsonl_data.append(data["id"])
            except json.JSONDecodeError as e:
                print(f"{file_path}第{line_num}行：JSON解析失败 - {e}，跳过")
            except Exception as e:
                print(f"{file_path}第{line_num}行：处理失败 - {e}，跳过")

    return jsonl_data


if __name__ == "__main__":
    # 固定随机种子（可选，如需复现结果则取消注释）
    # random.seed(42)
    sample_jsonl_files()