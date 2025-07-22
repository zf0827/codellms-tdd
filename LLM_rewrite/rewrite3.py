import json
import argparse
import os
import random
import string
import ast
from typing import List, Optional

def random_str(length=8):
    return ''.join(random.choices(string.ascii_letters, k=length))

class RenameTransformer(ast.NodeTransformer):
    def __init__(self):
        super().__init__()
        self.var_map = {}
        self.str_map = {}
        self.func_map = {}

    def _get_var(self, name):
        if name not in self.var_map:
            self.var_map[name] = random_str()
        return self.var_map[name]

    def _get_str(self, s):
        if s not in self.str_map:
            self.str_map[s] = random_str()
        return self.str_map[s]

    def _get_func(self, name):
        if name not in self.func_map:
            self.func_map[name] = random_str()
        return self.func_map[name]

    def visit_FunctionDef(self, node):
        node.name = self._get_func(node.name)
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node):
        node.name = self._get_func(node.name)
        self.generic_visit(node)
        return node

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Load, ast.Del)):
            node.id = self._get_var(node.id)
        return node

    def visit_arg(self, node):
        node.arg = self._get_var(node.arg)
        return node

    def visit_Attribute(self, node):
        self.generic_visit(node)
        # 不改写属性名（如obj.attr），只改变量名
        return node

    def visit_Str(self, node):
        node.s = self._get_str(node.s)
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = self._get_str(node.value)
        return node

def rewrite_code_with_ast(code_snippet: str) -> Optional[str]:
    try:
        tree = ast.parse(code_snippet)
        transformer = RenameTransformer()
        tree = transformer.visit(tree)
        ast.fix_missing_locations(tree)
        new_code = ast.unparse(tree)
        return new_code.strip()
    except Exception as e:
        print(f"AST改写失败: {e}")
        return None

def process_file(input_file: str, output_file: str):
    print(f"开始处理：{input_file} -> {output_file}")
    lines_processed = 0
    lines_failed = 0
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            for line_num, line in enumerate(infile, 1):
                try:
                    data = json.loads(line.strip())
                except json.JSONDecodeError:
                    print(f"警告：第{line_num}行不是有效JSON，已跳过：{line.strip()}")
                    continue
                if "input" not in data:
                    print(f"警告：第{line_num}行缺少'input'键，已跳过。")
                    continue
                original_code = data["input"]
                print(f"\n正在处理第{line_num}行...")
                rewritten_code = rewrite_code_with_ast(original_code)
                if rewritten_code:
                    data["input"] = rewritten_code
                    outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                    lines_processed += 1
                    print(f"第{line_num}行处理成功。")
                else:
                    print(f"错误：第{line_num}行重写失败。")
                    lines_failed += 1
    except FileNotFoundError:
        print(f"错误：未找到输入文件：{input_file}")
        exit(1)
    except Exception as e:
        print(f"文件处理过程中发生错误：{e}")
    finally:
        print(f"\n处理结束。")
        print(f"成功处理行数：{lines_processed}")
        print(f"失败行数：{lines_failed}")

def test_line(input_file: str, line_index: int = 1):
    print(f"--- 正在测试第{line_index}行 ---")
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            for current_index, line in enumerate(infile, 1):
                if current_index == line_index:
                    target_line = line
                    break
            else:
                print(f"错误：输入文件不足{line_index}行。")
                return
            try:
                data = json.loads(target_line.strip())
            except json.JSONDecodeError:
                print(f"错误：第{line_index}行不是有效的JSON: {target_line.strip()}")
                return
            if "input" not in data:
                print(f"错误：第{line_index}行JSON不包含'input'键。")
                return
            original_code = data["input"]
            print("\n--- 原始代码 ---")
            print(original_code)
            rewritten_code = rewrite_code_with_ast(original_code)
            if rewritten_code:
                print("\n--- 重写后代码 ---")
                print(rewritten_code)
            else:
                print("\n--- 重写失败 ---")
    except FileNotFoundError:
        print(f"错误：输入文件未找到: {input_file}")
    except Exception as e:
        print(f"测试过程中发生错误: {e}")

def main():
    parser = argparse.ArgumentParser(description="使用AST重写Python函数变量名和字符串为随机乱码。输入输出格式与SLICON_rewrite.py一致。")
    parser.add_argument("--input_file", help="输入JSONL文件路径。")
    parser.add_argument("--output_file", help="输出JSONL文件路径。")
    parser.add_argument("--test", nargs="?", const=1, type=int, help="测试模式：仅处理第i行并打印结果，i默认为1。")
    args = parser.parse_args()
    if args.test is not None:
        test_line(args.input_file, args.test)
    else:
        process_file(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
