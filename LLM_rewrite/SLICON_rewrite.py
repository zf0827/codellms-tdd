import requests
import json
import argparse
import os
import time
from typing import List, Dict, Optional

# 所有函数定义都在这里

def rewrite_code_with_retry(
    api_url: str,
    api_keys: List[str],
    prompt_template: str,
    code_snippet: str,
    model: str,
    max_retries: int = 3,
    initial_delay: float = 15.0,
) -> Optional[str]:
    """
    使用QWEN API重写代码，失败时自动切换API Key重试。

    参数：
        api_url: QWEN chat completions接口URL。
        api_keys: 可用API Key列表。
        prompt_template: 提示词模板（包含<Insert your function here>占位符）。
        code_snippet: 待重写的代码片段。
        model: QWEN模型名称。
        max_retries: 每个Key最大重试次数。
        initial_delay: 指数退避初始等待时间。

    返回：
        重写后的代码字符串，若全部失败则返回None。
    """
    last_exception = None
    for key_index, api_key in enumerate(api_keys):
        print(f"--- 正在尝试API Key #{key_index + 1} ---")
        current_delay = initial_delay
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        # 用模板替换<Insert your function here>为当前代码
        user_content = prompt_template.replace("<Insert your function here>", code_snippet)
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": user_content}
            ],
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(api_url, json=payload, headers=headers, timeout=60) # 增加超时
                response.raise_for_status() # 非2xx抛异常

                response_data = response.json()

                # 假设QWEN返回结构与OpenAI类似
                if response_data.get("choices") and len(response_data["choices"]) > 0:
                    message = response_data["choices"][0].get("message")
                    if message and message.get("content"):
                        rewritten_code = message["content"]
                        print(f"--- API Key #{key_index + 1} 成功 ---")
                        return rewritten_code.strip()
                    else:
                        print(f"警告：响应格式异常（Key #{key_index + 1}, 第{attempt + 1}次）：无content。")
                        last_exception = ValueError("响应格式异常")
                        break
                else:
                    print(f"警告：响应格式异常（Key #{key_index + 1}, 第{attempt + 1}次）：无choices。")
                    last_exception = ValueError("响应格式异常")
                    break

            except requests.exceptions.Timeout as e:
                last_exception = e
                print(f"警告：请求超时（Key #{key_index + 1}, 第{attempt + 1}/{max_retries}次）：{e}。{current_delay:.2f}秒后重试...")
                time.sleep(current_delay)
                current_delay *= 2
            except requests.exceptions.HTTPError as e:
                last_exception = e
                if e.response.status_code == 401: # 未授权
                    print(f"错误：API Key #{key_index + 1}认证失败（状态码{e.response.status_code}）。尝试下一个Key。")
                    break
                elif e.response.status_code == 429: # 速率限制
                    print(f"警告：速率限制（Key #{key_index + 1}, 第{attempt + 1}/{max_retries}次，状态码{e.response.status_code}）。{current_delay:.2f}秒后重试...")
                    time.sleep(current_delay)
                    current_delay *= 2
                elif 500 <= e.response.status_code < 600: # 服务器错误
                    print(f"警告：服务器错误（Key #{key_index + 1}, 第{attempt + 1}/{max_retries}次，状态码{e.response.status_code}）。{current_delay:.2f}秒后重试...")
                    time.sleep(current_delay)
                    current_delay *= 2
                else: # 其他HTTP错误
                    print(f"错误：HTTP错误（Key #{key_index + 1}, 状态码{e.response.status_code}）：{e}。尝试下一个Key。")
                    break
            except requests.exceptions.RequestException as e:
                last_exception = e
                print(f"警告：请求失败（Key #{key_index + 1}, 第{attempt + 1}/{max_retries}次）：{e}。{current_delay:.2f}秒后重试...")
                time.sleep(current_delay)
                current_delay *= 2
            except json.JSONDecodeError as e:
                last_exception = e
                print(f"错误：响应非JSON（Key #{key_index + 1}, 第{attempt + 1}次）：{e}。响应内容：{response.text[:100]}...")
                break
            except Exception as e:
                print(f"错误：API调用发生未知异常（Key #{key_index + 1}）：{e}")
                last_exception = e
                break
        else:
            print(f"警告：API Key #{key_index + 1}已达最大重试次数。")
            continue

    print(f"错误：所有API Key均失败。最后错误：{last_exception}")
    return None

# 处理整个文件，每行调用重写

def process_file(input_file: str, output_file: str, prompt_template: str, api_url: str, api_keys: List[str], model: str):
    """处理输入JSONL文件，重写后写入输出文件。"""
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

                rewritten_code = rewrite_code_with_retry(
                    api_url=api_url,
                    api_keys=api_keys,
                    prompt_template=prompt_template,
                    code_snippet=original_code,
                    model=model
                )

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

# 测试模式：只处理指定行

def test_line(input_file: str, prompt_template: str, api_url: str, api_keys: List[str], model: str, line_index: int = 1):
    """测试输入文件的第line_index行的重写过程。line_index从1开始。"""
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

            rewritten_code = rewrite_code_with_retry(
                api_url=api_url,
                api_keys=api_keys,
                prompt_template=prompt_template,
                code_snippet=original_code,
                model=model
            )

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
    # --- 配置区 ---
    # 请在此处填写你的QWEN API Key
    # API_KEYS = 
    API_URL = "https://api.siliconflow.cn/v1/chat/completions" # QWEN API接口
    DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct" # 默认QWEN模型

    parser = argparse.ArgumentParser(description="使用QWEN API重写Python函数。")
    parser.add_argument("--input_file", help="输入JSONL文件路径。")
    parser.add_argument("--output_file", help="输出JSONL文件路径。")
    parser.add_argument("--test", nargs="?", const=1, type=int, help="测试模式：仅处理第i行并打印结果，i默认为1。")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"QWEN模型名称（如{DEFAULT_MODEL}）。")
    parser.add_argument("--type", type=int, default=0, help="提示词类型：1=同义词改写，2=简单易懂，默认prompt.txt")

    args = parser.parse_args()

    # 根据type选择prompt模板
    if args.type == 1:
        prompt_path = os.path.join(os.path.dirname(__file__), "prompt1_random.txt")
    elif args.type == 2:
        prompt_path = os.path.join(os.path.dirname(__file__), "prompt2_simplify.txt")
    else:
        prompt_path = os.path.join(os.path.dirname(__file__), "prompt_err.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    api_keys = [key.strip() for key in API_KEYS if key.strip() and not key.startswith("YOUR_")]
    if not api_keys:
        print("错误：未找到有效API Key，请在脚本中填写你的QWEN API Key。")
        exit(1)

    api_url = API_URL
    model = args.model

    if args.test is not None:
        test_line(args.input_file, prompt_template, api_url, api_keys, model, args.test)
    else:
        process_file(args.input_file, args.output_file, prompt_template, api_url, api_keys, model)

if __name__ == "__main__":
    main()
