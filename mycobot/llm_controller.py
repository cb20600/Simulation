# llm_controller.py

import numpy as np
import json
import re
import difflib
from openai import OpenAI

def fuzzy_lookup(name: str, candidates: dict):
    """
    在 candidates 中模糊查找最接近 name 的 key，返回对应 value。
    """
    matches = difflib.get_close_matches(name.lower(), candidates.keys(), n=1, cutoff=0.6)
    if matches:
        print(f"[LLM] Fuzzy match: '{name}' → '{matches[0]}'")
        return candidates[matches[0]]
    return None

def extract_command_info(user_command: str, client: OpenAI):
    """
    使用 LLM 提取多个 objects 和一个 target。
    """
    prompt = f"""
You are a robot command parser. Extract all the fruit objects (e.g., apple, banana, orange) and the target bin (e.g., red bin, yellow bin) from the command.

Return this JSON format:
{{
  "objects": ["apple", "orange"],
  "target": "red bin"
}}

If the command is unclear (e.g., just says 'fruit' or 'bin'), return:
{{
  "error": "missing information",
  "ask": "Which fruits and which bin do you want to use? 
  "Fruits: apple, banana, orange, pear. Bins: red bin, yellow bin."
}}

Command: "{user_command}"
"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for parsing robot commands."},
            {"role": "user", "content": prompt.strip()}
        ],
        temperature=0.2
    )

    result_text = response.choices[0].message.content
    print("[LLM] Raw LLM output:", result_text)

    cleaned = re.sub(r"```(?:json)?", "", result_text).strip("`\n ")

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print("[LLM] JSON decode failed:", e)
        return {}

def run_llm_command(user_command, client: OpenAI, fruits, bins, grasp, place):
    print(f"\n[LLM] Command: {user_command}")
    parsed = extract_command_info(user_command, client)
    print("[LLM] Parsed:", parsed)

    if "error" in parsed:
        print("[LLM] Clarification needed:", parsed.get("ask"))
        return

    # 兼容字符串和数组格式
    obj_names = parsed.get("objects", [])
    if isinstance(obj_names, str):
        obj_names = [obj_names]

    bin_name = parsed.get("target", "").lower()

    # 映射表
    object_map = {
        "apple": fruits.get("apple"),
        "banana": fruits.get("banana"),
        "orange": fruits.get("orange"),
        "pear": fruits.get("pear"),
    }

    bin_map = {
        "red bin": bins.get("redBin"),
        "redbin": bins.get("redBin"),
        "yellow bin": bins.get("yellowBin"),
        "yellowbin": bins.get("yellowBin"),
    }

    # 匹配垃圾桶
    bin_entity = fuzzy_lookup(bin_name, bin_map)
    if bin_entity is None:
        print(f"[LLM] Unknown bin: {bin_name}")
        return
    target_pos = np.array(bin_entity.get_pos().tolist())

    # 执行每个水果的抓取和投放
    for obj_name in obj_names:
        obj_entity = fuzzy_lookup(obj_name.lower(), object_map)
        if obj_entity is None:
            print(f"[LLM] Unknown object: {obj_name}")
            continue

        print(f"[LLM] Executing: move '{obj_name}' to '{bin_name}'")
        grasp(obj_entity)
        place(target_pos)
