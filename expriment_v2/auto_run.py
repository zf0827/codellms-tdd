import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple
import os

def run_command(cmd: List[str], description: str) -> bool:
    """运行单个命令，返回是否成功"""
    print(f"\n{'='*50}")
    print(f"正在执行: {description}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("执行成功!")
        if result.stdout:
            print("输出:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"执行失败! 错误码: {e.returncode}")
        if e.stdout:
            print("标准输出:", e.stdout)
        if e.stderr:
            print("错误输出:", e.stderr)
        return False


def run_experiment_sequence(exp_name: str, attempt_id: int, size: int, lb: int, rb: int, sample_per_label: int, seed: int = 123) -> bool:
    """运行单个实验的完整流程"""
    print(f"\n开始实验: {exp_name}, attempt_id={attempt_id}")
    
    # 1. 运行 exp_maker.py
    cmd1 = [
        "python3", "exp_maker.py",
        "--exp_name", exp_name,
        "--size", str(size),
        "--lb", str(lb),
        "--rb", str(rb),
        "--seed", str(seed)
    ]
    if not run_command(cmd1, "创建基础数据集"):
        return False
    
    # 2. 运行 attempt_maker.py
    cmd2 = [
        "python3", "attempt_maker.py",
        "--exp_name", exp_name,
        "--attempt_id", str(attempt_id),
        "--sample_per_label", str(sample_per_label),
        # "--skip_existing"
    ]
    if not run_command(cmd2, "创建测试数据集"):
        return False
    
    # 3. 运行 result_maker.py
    cmd3 = [
        "python3", "result_maker.py",
        "--exp_name", exp_name,
        "--attempt_id", str(attempt_id),
        # "--skip_existing"
    ]
    if not run_command(cmd3, "生成评估结果"):
        return False
    
    print(f"实验 {exp_name} (attempt_id={attempt_id}) 完成!")
    return True


def main():
    # import subprocess

    # commands = [
    #     "python3 /home/yunxiang/work_june/lixian/temp.py"
    # ]
    # lixian_dir = "/home/yunxiang/work_june/lixian"
    # for cmd in commands:
    #     print(f"正在执行: {cmd}")
    #     try:
    #         ret = subprocess.run(cmd, shell=True, cwd=lixian_dir)
    #         if ret.returncode != 0:
    #             print(f"命令执行失败: {cmd}")
    #             break
    #     except Exception as e:
    #         print(f"执行过程中发生异常: {e}")
    #         break
    # return
    # run_experiment_sequence("exp1a", 1, 100, 100, 200, 100, 123)
    # run_experiment_sequence("exp2a", 1, 250, 200, 300, 250, 123)
    # run_experiment_sequence("exp3a", 1, 350, 300, 400, 350, 123)
    # run_experiment_sequence("exp4a", 1, 350, 400, 500, 350, 123)
    # run_experiment_sequence("exp5a", 1, 250, 500, 600, 250, 123)
    # run_experiment_sequence("exp1b", 1, 100, 100, 200, 100, 66)
    # run_experiment_sequence("exp2b", 1, 250, 200, 300, 250, 66)
    # run_experiment_sequence("exp3b", 1, 350, 300, 400, 350, 66)
    # run_experiment_sequence("exp4b", 1, 350, 400, 500, 350, 66)
    # run_experiment_sequence("exp5b", 1, 250, 500, 600, 250, 66)

    # run_experiment_sequence("exp1c", 1, 100, 100, 200, 100, 32)
    # run_experiment_sequence("exp2c", 1, 250, 200, 300, 250, 32)
    # run_experiment_sequence("exp3c", 1, 350, 300, 400, 350, 32)
    # run_experiment_sequence("exp4c", 1, 350, 400, 500, 350, 32)
    # run_experiment_sequence("exp5c", 1, 250, 500, 600, 250, 32)
    # run_experiment_sequence("exp1d", 1, 100, 100, 200, 100, 77)
    # run_experiment_sequence("exp2d", 1, 250, 200, 300, 250, 77)
    # run_experiment_sequence("exp3d", 1, 350, 300, 400, 350, 77)
    # run_experiment_sequence("exp4d", 1, 350, 400, 500, 350, 77)
    # run_experiment_sequence("exp5d", 1, 250, 500, 600, 250, 77)

    run_experiment_sequence("exp_len_100a", 1, 100, 100, 200, 100, 88)
    # run_experiment_sequence("exp_len_200a", 1, 100, 200, 250, 100, 88)
    run_experiment_sequence("exp_len_300a", 1, 100, 350, 400, 100, 88)
    run_experiment_sequence("exp_len_400a", 1, 100, 400, 450, 100, 88)
    run_experiment_sequence("exp_len_500a", 1, 100, 550, 600, 100, 88)
    run_experiment_sequence("exp_len_600a", 1, 100, 600, 700, 100, 88)
    run_experiment_sequence("exp_len_100b", 1, 100, 100, 200, 100, 66)
    run_experiment_sequence("exp_len_200b", 1, 100, 200, 250, 100, 66)
    run_experiment_sequence("exp_len_300b", 1, 100, 350, 400, 100, 66)
    run_experiment_sequence("exp_len_400b", 1, 100, 400, 450, 100, 66)
    run_experiment_sequence("exp_len_500b", 1, 100, 550, 600, 100, 66)
    run_experiment_sequence("exp_len_600b", 1, 100, 600, 700, 100, 66)
    run_experiment_sequence("exp_len_100c", 1, 100, 100, 200, 100, 32)
    run_experiment_sequence("exp_len_200c", 1, 100, 200, 250, 100, 32)
    run_experiment_sequence("exp_len_300c", 1, 100, 350, 400, 100, 32)
    run_experiment_sequence("exp_len_400c", 1, 100, 400, 450, 100, 32)
    run_experiment_sequence("exp_len_500c", 1, 100, 550, 600, 100, 32)
    run_experiment_sequence("exp_len_600c", 1, 100, 600, 700, 100, 32)
    run_experiment_sequence("exp_len_100d", 1, 100, 100, 200, 100, 77)
    run_experiment_sequence("exp_len_200d", 1, 100, 200, 250, 100, 77)
    run_experiment_sequence("exp_len_300d", 1, 100, 350, 400, 100, 77)
    run_experiment_sequence("exp_len_400d", 1, 100, 400, 450, 100, 77)
    run_experiment_sequence("exp_len_500d", 1, 100, 550, 600, 100, 77)
    run_experiment_sequence("exp_len_600d", 1, 100, 600, 700, 100, 77)

    
    # run_command(["python3", "exp_maker.py", "--exp_name", "exp3a", "--size", "350", "--lb", "300", "--rb", "400", "--seed", "123"], "运行exp_maker.py")
    # run_command(["python3", "exp_maker.py", "--exp_name", "exp3b", "--size", "350", "--lb", "300", "--rb", "400", "--seed", "66"], "运行exp_maker.py")
    # run_command(["python3", "exp_maker.py", "--exp_name", "exp3c", "--size", "350", "--lb", "300", "--rb", "400", "--seed", "32"], "运行exp_maker.py")
    # run_command(["python3", "exp_maker.py", "--exp_name", "exp3d", "--size", "350", "--lb", "300", "--rb", "400", "--seed", "77"], "运行exp_maker.py")
    # run_command(["python3", "attempt_maker.py", "--exp_name", "exp3a", "--attempt_id", "1", "--sample_per_label", "350"], "运行attempt_maker.py")
    # run_command(["python3", "attempt_maker.py", "--exp_name", "exp3b", "--attempt_id", "1", "--sample_per_label", "350", "--skip_existing"], "运行attempt_maker.py")
    # run_command(["python3", "attempt_maker.py", "--exp_name", "exp3c", "--attempt_id", "1", "--sample_per_label", "350", "--skip_existing"], "运行attempt_maker.py")
    # run_command(["python3", "attempt_maker.py", "--exp_name", "exp3d", "--attempt_id", "1", "--sample_per_label", "350", "--skip_existing"], "运行attempt_maker.py")
    # run_command(["python3", "result_maker.py", "--exp_name", "exp3a", "--attempt_id", "1" ], "运行result_maker.py")
    # run_command(["python3", "result_maker.py", "--exp_name", "exp3b", "--attempt_id", "1" ], "运行result_maker.py")
    # run_command(["python3", "result_maker.py", "--exp_name", "exp3c", "--attempt_id", "1" ], "运行result_maker.py")
    # run_command(["python3", "result_maker.py", "--exp_name", "exp3d", "--attempt_id", "1" ], "运行result_maker.py")

    return
    parser = argparse.ArgumentParser(description="自动运行实验流程")
    parser.add_argument("--experiments", nargs="+", required=True, 
                       help="实验配置，格式: exp_name:attempt_id:size:lb:rb:sample_per_label")
    parser.add_argument("--seed", type=int, default=123, help="随机种子")
    

    args = parser.parse_args()
    
    # 解析实验配置
    experiments: List[Tuple[str, int, int, int, int, int]] = []
    for exp_config in args.experiments:
        try:
            parts = exp_config.split(":")
            if len(parts) != 6:
                raise ValueError(f"配置格式错误: {exp_config}")
            
            exp_name = parts[0]
            attempt_id = int(parts[1])
            size = int(parts[2])
            lb = int(parts[3])
            rb = int(parts[4])
            sample_per_label = int(parts[5])
            
            experiments.append((exp_name, attempt_id, size, lb, rb, sample_per_label))
        except ValueError as e:
            print(f"解析配置失败: {exp_config}, 错误: {e}")
            sys.exit(1)
    
    print(f"将运行 {len(experiments)} 个实验:")
    for i, (exp_name, attempt_id, size, lb, rb, sample_per_label) in enumerate(experiments, 1):
        print(f"  {i}. {exp_name} (attempt_id={attempt_id}, size={size}, lb={lb}, rb={rb}, sample_per_label={sample_per_label})")
    
    # 运行所有实验
    success_count = 0
    for exp_name, attempt_id, size, lb, rb, sample_per_label in experiments:
        if run_experiment_sequence(exp_name, attempt_id, size, lb, rb, sample_per_label, args.seed):
            success_count += 1
        else:
            print(f"实验 {exp_name} (attempt_id={attempt_id}) 失败，继续下一个...")
    
    print(f"\n{'='*50}")
    print(f"所有实验完成! 成功: {success_count}/{len(experiments)}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
