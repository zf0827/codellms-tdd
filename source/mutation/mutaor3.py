import ast
import json # 为 make_dataset 添加
import os   # 为 make_dataset 添加
import argparse
import sys
# import astor  # 用于将AST转换回源代码 (pip install astor)
              # 如果使用 Python 3.9+，可以考虑内置的 ast.unparse
from pattern2 import apply_perturbation_if_and_true
from pattern4 import apply_perturbation_equality_to_not_in_equality
from pattern1 import apply_perturbation_assignment_temp_var
from pattern3 import apply_perturbation_print_log, apply_perturbation_try_except_reraise

# 扰动策略注册表
# 键是策略的唯一名称，值是应用该策略的函数
PERTURBATION_REGISTRY = {
    "if_condition_and_true": apply_perturbation_if_and_true,
    # 在这里添加其他模式的扰动函数，例如：
    "assignment_to_temp_var": apply_perturbation_assignment_temp_var,
    "print_log_after_assignment": apply_perturbation_print_log,
    "try_except_reraise_wrapper": apply_perturbation_try_except_reraise,
    "equality_to_not_in_equality": apply_perturbation_equality_to_not_in_equality,
}

# 默认的扰动幅度配置，用于 make_dataset
DEFAULT_MUTATION_THRESHOLDS = {
    "if_condition_and_true": 0.5,
    "assignment_to_temp_var": 0.5,
    "print_log_after_assignment": 0.3,
    "try_except_reraise_wrapper": 0.5,
    "equality_to_not_in_equality": 0.5
}

def perturb_python_code(code_string, perturbation_configs):
    """
    对Python代码字符串应用一系列指定的扰动。

    Args:
        code_string (str): 要扰动的Python源代码。
        perturbation_configs (list): 一个配置列表，其中每个元素指定一个扰动及其参数。
                                     示例: [
                                         {"name": "if_condition_and_true", "threshold_ratio": 0.5},
                                         # {"name": "assignment_to_temp_var", "threshold_ratio": 1.0}
                                     ]

    Returns:
        str: 扰动后的Python代码字符串。
    """
    try:
        current_ast = ast.parse(code_string)
    except SyntaxError as e:
        print(f"代码解析错误: {e} 对于代码: \n{code_string[:200]}...") # 打印部分代码帮助定位
        return code_string 

    for config in perturbation_configs:
        strategy_name = config.get("name")
        if not strategy_name:
            # print(f"警告: 配置项缺少 'name': {config}") # 在批量处理时可能过于冗余
            continue
        strategy_func = PERTURBATION_REGISTRY.get(strategy_name)
        if not strategy_func:
            # print(f"警告: 未找到扰动策略 '{strategy_name}'。跳过。")
            continue
        params = {key: value for key, value in config.items() if key != "name"}
        # print(f"应用扰动: {strategy_name} 使用参数: {params}") # 在批量处理时可能过于冗余
        try:
            current_ast = strategy_func(current_ast, **params)
        except Exception as e:
            print(f"警告: 应用策略 '{strategy_name}' 时出错: {e} 对于代码: \n{code_string[:200]}...")
            # 在批量处理时，遇到单个策略错误，可以选择继续处理AST，而不是返回原始代码
            # return code_string 

    try:
        perturbed_code = ast.unparse(current_ast)
    except Exception as e:
        print(f"将AST转换回源代码时出错: {e}")
        return code_string 
    return perturbed_code

# --- Mutate Function (formerly overall_test) ---

def mutate3(code_snippet, perturbation_thresholds=None, similarity=None):
    """
    对给定的代码片段应用所有指定的扰动策略，并返回扰动后的代码。
    此函数主要用于数据集生成，因此错误处理和日志记录比之前的 overall_test 更简洁。

    Args:
        code_snippet (str): 要扰动的Python源代码片段。
        perturbation_thresholds (dict): 字典，键是策略名，值是 threshold_ratio。
        similarity (float): 目标相似度（0~1），如果提供则根据相似度自动生成扰动阈值。
    Returns:
        str: 扰动后的Python代码字符串。
    """
    if similarity is not None:
        perturbation_thresholds = compute_mutation_ratios_by_similarity(code_snippet, similarity)
    elif perturbation_thresholds is None:
        perturbation_thresholds = DEFAULT_MUTATION_THRESHOLDS
    configs_to_apply = []
    for strategy_name, threshold in perturbation_thresholds.items():
        if strategy_name in PERTURBATION_REGISTRY and isinstance(threshold, (float, int)) and 0 < threshold <= 1.0:
            configs_to_apply.append({"name": strategy_name, "threshold_ratio": float(threshold)})
        # 可以选择性地为无效的策略名或阈值添加日志，但在批量处理中通常省略
    
    if not configs_to_apply:
        # print("没有配置有效的扰动策略以供应用。返回原始代码。")
        return code_snippet

    # 重置计数器（如果需要，特定于某些模式）
    if "assignment_to_temp_var" in perturbation_thresholds and perturbation_thresholds["assignment_to_temp_var"] > 0:
            from pattern1 import AssignmentToTempVarTransformer
            AssignmentToTempVarTransformer.reset_counter()
    # 其他模式的计数器重置逻辑可以类似地添加

    return perturb_python_code(code_snippet, configs_to_apply)

# --- Dataset Creation Function ---

def process_file(input_file, output_file, mutation_thresholds=None, similarity=None):
    """
    批量扰动文件。支持通过 similarity 参数自动生成扰动阈值。
    """
    processed_count = 0
    skipped_count = 0
    print(f"\n--- 批量扰动: 处理文件 '{input_file}' -> '{output_file}' ---")
    if similarity is not None:
        print(f"使用相似度参数: {similarity}，将自动为每条代码生成扰动阈值。")
    else:
        thresholds_to_use = mutation_thresholds if mutation_thresholds is not None else DEFAULT_MUTATION_THRESHOLDS
        print(f"使用扰动阈值: {thresholds_to_use}")
    try:
        with open(input_file, "r", encoding="utf-8") as infile, \
             open(output_file, "w", encoding="utf-8") as outfile:
            for line_num, line in enumerate(infile, 1):
                line_content = line.strip()
                if not line_content:
                    skipped_count +=1
                    continue
                try:
                    data = json.loads(line_content)
                    original_code = data.get("input")
                    label = data.get("label")
                    # 针对每条代码，若 similarity 参数存在，则动态生成阈值
                    if original_code is not None and label is not None:
                        try:
                            if similarity is not None:
                                thresholds_to_use = compute_mutation_ratios_by_similarity(original_code, similarity)
                            elif mutation_thresholds is not None:
                                thresholds_to_use = mutation_thresholds
                            else:
                                thresholds_to_use = DEFAULT_MUTATION_THRESHOLDS
                            mutated_code = mutate3(original_code, thresholds_to_use)
                        except SyntaxError as e:
                            print(f"警告: 第 {line_num} 行代码存在语法错误: {e}，已跳过。")
                            skipped_count += 1
                            continue
                        except Exception as e:
                            print(f"警告: 第 {line_num} 行代码扰动时发生异常: {type(e).__name__} - {e}，已跳过。")
                            skipped_count += 1
                            continue
                        # 创建新数据点，保留原始数据点除input和label外的所有属性
                        new_data = data.copy()
                        new_data["input"] = mutated_code
                        new_data["label"] = label
                        outfile.write(json.dumps(new_data, ensure_ascii=False) + '\n')
                        processed_count += 1
                    else:
                        print(f"警告: 第 {line_num} 行缺少 'input' 或 'label'，已跳过。")
                        skipped_count += 1
                except json.JSONDecodeError as e:
                    print(f"警告: 解析第 {line_num} 行JSON时出错: {e}，已跳过。")
                    skipped_count += 1
        print(f"处理完成。成功处理 {processed_count} 条记录，跳过 {skipped_count} 条记录。")
        if processed_count > 0:
            print(f"结果已写入 '{output_file}'")
    except Exception as e:
        print(f"处理文件时发生严重错误: {type(e).__name__} - {e}")

def test_line(input_file, line_index=1, mutation_thresholds=None, similarity=None):
    """
    单行扰动测试。支持通过 similarity 参数自动生成扰动阈值。
    """
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if not lines:
            print(f"错误: 文件 '{input_file}' 为空。")
            return
        if not (1 <= line_index <= len(lines)):
            print(f"错误: 行号 {line_index} 超出文件 '{input_file}' 的范围 (共 {len(lines)} 行)。")
            return
        line_content = lines[line_index-1].strip()
        if not line_content:
            print(f"错误: 文件 '{input_file}' 的第 {line_index} 行为空。")
            return
        data = json.loads(line_content)
        original_code = data.get("input")
        if original_code is None:
            print(f"错误: 文件 '{input_file}' 第 {line_index} 行的JSON中未找到 'input' 字段或其值为 null。")
            return
        print(f"\n--- 单行扰动: 处理 '{input_file}' 第 {line_index} 行 ---")
        print("原始代码:")
        print(original_code)
        # 根据 similarity 参数动态生成阈值
        if similarity is not None:
            thresholds_to_use = compute_mutation_ratios_by_similarity(original_code, similarity)
            print(f"使用相似度参数: {similarity}，自动生成扰动阈值: {thresholds_to_use}")
        elif mutation_thresholds is not None:
            thresholds_to_use = mutation_thresholds
            print(f"使用扰动阈值: {thresholds_to_use}")
        else:
            thresholds_to_use = DEFAULT_MUTATION_THRESHOLDS
            print(f"使用默认扰动阈值: {thresholds_to_use}")
        mutated_code = mutate3(original_code, thresholds_to_use)
        print("\n扰动后代码:")
        print(mutated_code)
    except FileNotFoundError:
        print(f"错误: 文件 '{input_file}' 未找到。")
    except json.JSONDecodeError as e:
        print(f"错误: 解析 '{input_file}' 第 {line_index} 行的JSON时出错: {e}")
    except Exception as e:
        print(f"处理单行测试时发生未知错误: {type(e).__name__} - {e}")

def main():
    parser = argparse.ArgumentParser(description="批量/单行扰动Python代码，输入输出为JSONL。")
    parser.add_argument("--input_file", required=True, help="输入JSONL文件路径。")
    parser.add_argument("--output_file", help="输出JSONL文件路径。")
    parser.add_argument("--test", nargs="?", const=1, type=int, help="测试模式：仅处理第i行并打印结果，i默认为1。")
    parser.add_argument("--similarity", nargs="?", const=0.8, type=float, help="相似度参数，默认为0.8。")
    # 可扩展：parser.add_argument("--thresholds", ...)
    args = parser.parse_args()

    # 自定义阈值，暂时不用
    mutation_thresholds = {
        "if_condition_and_true": 1,
        "assignment_to_temp_var": 0.8,
        "print_log_after_assignment": 0.5,
        "try_except_reraise_wrapper": 0.7,
        "equality_to_not_in_equality": 0.5
    }

    # 相似度参数
    similarity = args.similarity

    if args.test is not None:
        test_line(args.input_file, args.test, mutation_thresholds, similarity)
    else:
        if not args.output_file:
            print("错误: 批量处理时必须指定 --output_file。")
            sys.exit(1)
        process_file(args.input_file, args.output_file, mutation_thresholds, similarity)

def compute_mutation_ratios_by_similarity(code_string, target_similarity):
    import ast
    from pattern1 import count_perturbation_candidates as count_assign
    from pattern2 import count_perturbation_candidates as count_if
    from pattern3 import count_perturbation_candidates_log, count_perturbation_candidates_try
    from pattern4 import count_perturbation_candidates as count_eq

    ast_root = ast.parse(code_string)
    total_lines = len(code_string.splitlines())
    n_if = count_if(ast_root)
    n_assign = count_assign(ast_root)
    n_log = count_perturbation_candidates_log(ast_root)
    n_try = count_perturbation_candidates_try(ast_root)
    n_eq = count_eq(ast_root)
    c_if, c_assign, c_log, c_try, c_eq = 1, 2, 1, 3, 1
    max_contrib = n_if*c_if + n_assign*c_assign + n_log*c_log + n_try*c_try + n_eq*c_eq
    if max_contrib == 0 or total_lines == 0:
        return {k: 0.0 for k in [
            "if_condition_and_true", "assignment_to_temp_var", "print_log_after_assignment",
            "try_except_reraise_wrapper", "equality_to_not_in_equality"
        ]}
    target_diff_lines = int(target_similarity * total_lines)
    per_type = max(1, target_diff_lines // 5)
    r_if = min(1.0, per_type / n_if) if n_if else 0.0
    r_assign = min(1.0, per_type / n_assign) if n_assign else 0.0
    r_log = min(1.0, per_type / n_log) if n_log else 0.0
    r_try = min(1.0, per_type / n_try) if n_try else 0.0
    r_eq = min(1.0, per_type / n_eq) if n_eq else 0.0
    return {
        "if_condition_and_true": r_if,
        "assignment_to_temp_var": r_assign,
        "print_log_after_assignment": r_log,
        "try_except_reraise_wrapper": r_try,
        "equality_to_not_in_equality": r_eq
    }

if __name__ == "__main__":
    main()