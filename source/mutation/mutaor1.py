import random
import subprocess
import json
import os
import argparse
import sys

def generate_random_style():
    indent_width = random.choice([2, 4, 8])  # 随机选择缩进宽度
    column_limit = random.choice([60, 80, 100, 120])  # 随机选择列限制
    align_closing_bracket_with_visual_indent = random.choice([True, False])  # 是否将右括号与视觉缩进对齐
    blank_lines_around_top_level_definition = random.choice([1, 2])  # 顶级定义周围的空行数
    space_inside_brackets = random.choice([True, False])  # 是否在括号内添加空格
    split_before_logical_operator = random.choice([True, False])  # 是否在逻辑运算符前换行
    use_tabs = random.choice([True, False])  # 是否使用制表符而不是空格

    style = f"""
    [style]
    based_on_style = pep8
    indent_width = {indent_width}
    column_limit = {column_limit}
    align_closing_bracket_with_visual_indent = {align_closing_bracket_with_visual_indent}
    blank_lines_around_top_level_definition = {blank_lines_around_top_level_definition}
    space_inside_brackets = {space_inside_brackets}
    split_before_logical_operator = {split_before_logical_operator}
    use_tabs = {use_tabs}
    """
    return style

def format_code_with_yapf(code):
    style = generate_random_style()
    style_file_name = ".yapf_style_temp.ini"
    temp_code_file_name = "temp_code_for_yapf.py"

    try:
        with open(style_file_name, "w") as f:
            f.write(style)
        
        with open(temp_code_file_name, "w") as f:
            f.write(code)

        subprocess.run(["yapf", "--in-place", temp_code_file_name, f"--style={style_file_name}"], check=True)

        with open(temp_code_file_name, "r") as f:
            formatted_code = f.read()
        
        return formatted_code
    finally:
        try:
            if os.path.exists(style_file_name):
                os.remove(style_file_name)
            if os.path.exists(temp_code_file_name):
                os.remove(temp_code_file_name)
        except OSError as e:
            print(f"警告: 清理临时文件 {style_file_name} 或 {temp_code_file_name} 时出错: {e}")

def process_file(input_file, output_file):
    processed_count = 0
    skipped_count = 0
    try:
        with open(input_file, "r", encoding="utf-8") as infile, \
             open(output_file, "w", encoding="utf-8") as outfile:
            for line_num, line in enumerate(infile, 1):
                line_content = line.strip()
                if not line_content:
                    skipped_count += 1
                    continue
                try:
                    data = json.loads(line_content)
                    original_code = data.get("input")
                    label = data.get("label")
                    if original_code is not None and label is not None:
                        mutated_code = format_code_with_yapf(original_code)
                        new_data = {"input": mutated_code, "label": label}
                        outfile.write(json.dumps(new_data, ensure_ascii=False) + '\n')
                        processed_count += 1
                    else:
                        print(f"警告: '{input_file}' 第 {line_num} 行缺少 'input' 或 'label' 字段 (或其值为 null)，已跳过。")
                        skipped_count += 1
                except json.JSONDecodeError as e:
                    print(f"警告: 解析 '{input_file}' 第 {line_num} 行的JSON时出错: {e}，已跳过。")
                    skipped_count += 1
                except subprocess.CalledProcessError as e:
                    print(f"警告: 调用yapf格式化第 {line_num} 行的代码时出错: {e}，已跳过。")
                    skipped_count += 1
                except Exception as e:
                    print(f"警告: 处理 '{input_file}' 第 {line_num} 行时发生未知错误: {type(e).__name__} - {e}，已跳过。")
                    skipped_count += 1
        print(f"处理完成。成功处理 {processed_count} 条记录，跳过 {skipped_count} 条记录。")
        if processed_count > 0:
            print(f"结果已写入 '{output_file}'")
    except Exception as e:
        print(f"处理文件时发生未知错误: {type(e).__name__} - {e}")

def test_line(input_file, line_index=1):
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
        print(f"--- 从 '{input_file}' 第 {line_index} 行读取 ---")
        print("原始代码片段:")
        print(original_code)
        mutated_code = format_code_with_yapf(original_code)
        print("\n调整后代码片段:")
        print(mutated_code)
    except FileNotFoundError:
        print(f"错误: 文件 '{input_file}' 未找到。")
    except json.JSONDecodeError as e:
        print(f"错误: 解析 '{input_file}' 第 {line_index} 行的JSON时出错: {e}")
    except subprocess.CalledProcessError as e:
        print(f"错误: 调用yapf格式化代码时出错: {e}")
    except Exception as e:
        print(f"处理单行测试时发生未知错误: {type(e).__name__} - {e}")

def main():
    parser = argparse.ArgumentParser(description="使用yapf随机风格批量/单行格式化Python代码。输入输出为JSONL。")
    parser.add_argument("--input_file", required=True, help="输入JSONL文件路径。")
    parser.add_argument("--output_file", help="输出JSONL文件路径。")
    parser.add_argument("--test", nargs="?", const=1, type=int, help="测试模式：仅处理第i行并打印结果，i默认为1。")
    args = parser.parse_args()

    if args.test is not None:
        test_line(args.input_file, args.test)
    else:
        if not args.output_file:
            print("错误: 批量处理时必须指定 --output_file。")
            sys.exit(1)
        process_file(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
