import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="批量生成三种改写结果，支持单行测试。")
    parser.add_argument("--input_file", required=True, help="输入JSONL文件路径。")
    parser.add_argument("--output_file", help="输出文件基础名（不带后缀）。")
    parser.add_argument("--test", nargs="?", const=1, type=int, help="测试模式：仅处理第i行并打印结果，i默认为1。")
    args = parser.parse_args()

    base_out = args.output_file
    input_file = args.input_file
    test_arg = args.test

    # 路径
    slicon = os.path.join(os.path.dirname(__file__), "SLICON_rewrite.py")
    rewrite3 = os.path.join(os.path.dirname(__file__), "rewrite3.py")

    # 定义命令及其对应的输出文件
    cmds_with_outs = []
    if test_arg is not None:
        # 单行测试，无输出文件
        cmds_with_outs.append((f"python3 {slicon} --input_file {input_file} --test {test_arg} --type 1", None))
        cmds_with_outs.append((f"python3 {slicon} --input_file {input_file} --test {test_arg} --type 2", None))
        cmds_with_outs.append((f"python3 {rewrite3} --input_file {input_file} --test {test_arg}", None))
    else:
        # 批量处理，指定输出文件
        out1 = f"{base_out}_1.jsonl"
        out2 = f"{base_out}_2.jsonl"
        out3 = f"{base_out}_3.jsonl"
        cmds_with_outs.append((f"python3 {slicon} --input_file {input_file} --output_file {out1} --type 1", out1))
        cmds_with_outs.append((f"python3 {slicon} --input_file {input_file} --output_file {out2} --type 2", out2))
        cmds_with_outs.append((f"python3 {rewrite3} --input_file {input_file} --output_file {out3}", out3))

    for cmd, out_file in cmds_with_outs:
        if out_file is not None and os.path.exists(out_file):
            print(f"跳过命令（输出文件已存在）: {cmd}")
            continue
        print(f"运行命令: {cmd}")
        ret = os.system(cmd)
        if ret != 0:
            print(f"命令执行失败: {cmd}")
            sys.exit(1)

if __name__ == "__main__":
    main()
