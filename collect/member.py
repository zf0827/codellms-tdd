import json
import time
from datetime import datetime, timezone
from datasets import load_dataset
import random
import os
import boto3
from smart_open import open as smart_open

# --- 配置 ---
OUTPUT_FILENAME = "cpp_samples.jsonl"
NUM_SAMPLES = 3000  # 要提取的样本总数

# --- AWS S3 配置 ---
try:
    session = boto3.Session(
        aws_access_key_id="AKIA4RCAOH2WQMJNVKVZ",
        aws_secret_access_key="M80VdKg39Piqr5pRokLhurhq03owHCUB45isWPrn"
    )
    s3 = session.client("s3")
    print("AWS Session 和 S3 Client 初始化成功。")
except Exception as e:
    print(f"初始化 AWS Session 或 S3 Client 时出错: {e}")
    exit(1)

# --- 数据收集列表 ---
collected_data = []
samples_count = 0

# --- 处理 the-stack-v2 ---
print("开始处理 the-stack-v2 数据集...")
try:
    ds_stack_v2 = load_dataset("bigcode/the-stack-v2", "C++", split="train", streaming=True)
    print("成功加载 the-stack-v2 数据流。")

    def download_contents(blob_id, src_encoding):
        """从 S3 下载并解码文件内容"""
        s3_url = f"s3://softwareheritage/content/{blob_id}"
        try:
            with smart_open(s3_url, "rb", compression=".gz", transport_params={"client": s3}) as fin:
                # 尝试使用指定的编码解码，如果失败则尝试 utf-8，最后尝试忽略错误
                try:
                    content = fin.read().decode(src_encoding)
                except (UnicodeDecodeError, LookupError): # LookupError for invalid encoding name
                    try:
                        # 回退到 utf-8
                        fin.seek(0) # 需要重置文件指针
                        content = fin.read().decode('utf-8')
                    except UnicodeDecodeError:
                        # 如果 utf-8 也失败，忽略错误
                        fin.seek(0)
                        content = fin.read().decode('utf-8', errors='ignore')
            return {"content": content}
        except Exception as e:
            return {"content": None} # 返回 None 表示失败

    processed_count = 0
    start_time = time.time()
    s3_download_success_count = 0
    s3_download_failure_count = 0

    for example in ds_stack_v2:
        if samples_count >= NUM_SAMPLES:
            print(f"已收集足够的样本 ({samples_count} 条)。")
            break # 如果已经收集够了，提前退出循环

        processed_count += 1
        if processed_count % 10000 == 0:
            elapsed_time = time.time() - start_time
            print(f"已处理 the-stack-v2 {processed_count} 条记录。已收集: {samples_count} 条样本。"
                  f"S3下载统计: 成功={s3_download_success_count}, 失败={s3_download_failure_count}。"
                  f"耗时: {elapsed_time:.2f} 秒")

        try:
            blob_id = example.get("blob_id")
            if not blob_id:
                continue
            src_encoding = example.get("src_encoding", "utf-8") # 获取编码，默认为 utf-8

            # 调用函数从 S3 下载内容
            download_result = download_contents(blob_id, src_encoding)
            code_content = download_result["content"]

            if code_content is None:
                # 下载失败
                s3_download_failure_count += 1
                continue
            else:
                # 下载成功
                s3_download_success_count += 1

            # 获取访问日期（如果有）
            visit_date_str = None
            try:
                visit_date_obj = example.get("visit_date")
                if hasattr(visit_date_obj, 'to_pydatetime'):
                    visit_datetime = visit_date_obj.to_pydatetime()
                    if visit_datetime.tzinfo is None:
                        visit_datetime = visit_datetime.replace(tzinfo=timezone.utc)
                    visit_date_str = visit_datetime.isoformat().replace('+00:00', 'Z')
            except:
                pass

            collected_data.append({
                "content": code_content,
                "timestamp": visit_date_str,
                "repo_name": example.get("repo_name", ""),
                "path": example.get("path", ""),
                "source": "the-stack-v2"
            })
            samples_count += 1

        except Exception as e:
            continue

    print(f"处理完成。S3 下载统计: 成功={s3_download_success_count}, 失败={s3_download_failure_count}。")

except Exception as load_err:
    print(f"加载或处理 the-stack-v2 数据集时发生错误: {load_err}")

# --- 保存到 JSONL 文件 ---
print(f"总共收集到 {len(collected_data)} 条代码样本。")

if collected_data:
    print(f"正在将数据写入 {OUTPUT_FILENAME}...")
    # 随机打乱数据
    random.shuffle(collected_data)
    try:
        with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
            for item in collected_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"数据成功写入 {OUTPUT_FILENAME}")
    except IOError as e:
        print(f"写入文件时出错: {e}")
else:
    print("没有收集到任何数据，未创建输出文件。")

print("处理完成。")
