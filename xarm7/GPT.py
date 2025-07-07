import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import re

# 初始化 API
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def parser(user_input):
    # LLM 理解指令并输出任务步骤（自然语言描述）
    messages = [
        {
            "role": "system",
            "content": (
                "You are a robot assistant that understands natural language instructions.\n"
                "Your job is to interpret the user's task command and output a step-by-step subtask plan.\n"
                "Each step should describe clearly:\n"
                "- what to do\n"
                "- which objects are involved\n"
                "- what the goal is\n"
                "- any spatial relations or constraints\n\n"
                "Output should be plain text only, no JSON or code formatting."
            )
        },
        {
            "role": "user",
            "content": user_input
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.4
    )

    print("\n🧠 Subtask Plan:")
    print(response.choices[0].message.content)

    return response.choices[0].message.content


def trajectory_plan(subtask_txt_path="llm_subtasks.txt", grasp_json_path="grasp_infos.json", save_json_path="trajectory_plan.json"):
    """
    基于任务拆解和抓取信息，调用 LLM 生成自然语言描述轨迹，并提取关键字段保存为 JSON 文件。

    返回:
        llm_output (str): LLM 原始自然语言轨迹描述
        parsed_steps (list): 结构化轨迹数据
    """
    # 读取任务描述
    with open(subtask_txt_path, "r") as f:
        subtask_text = f.read()

    # 读取抓取信息
    with open(grasp_json_path, "r") as f:
        grasp_data = json.load(f)

    # 提问 LLM
    messages = [
        {
            "role": "system",
            "content": (
                "You are a robot trajectory planner. Based on the user's task breakdown and the sensed object information, "
                "generate a detailed motion plan for the robot arm to follow.\n"
                "Each action should include:\n"
                "- A step index\n"
                "- Target 3D position (x, y, z)\n"
                "- Target orientation (quaternion: x, y, z, w)\n"
                "- Gripper action: open or close or maintain grip\n"
                "- Gripper value (in percent if open/close)\n"
                "- Description of what this step does\n\n"
                "Output should be clearly numbered natural language steps (for human reading),"
                "but ensure each numeric field (position, quaternion, gripper) is in strict format to allow extraction."
            )
        },
        {
            "role": "user",
            "content": f"Task description:\n{subtask_text.strip()}\n\nSensed grasp data:\n{json.dumps(grasp_data, indent=2)}"
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.4
    )

    llm_output = response.choices[0].message.content
    print("\n🧠 LLM Trajectory Plan:\n")
    print(llm_output)

    # 保存 LLM 输出为文本
    with open("trajectory_plan_raw.txt", "w") as f:
        f.write(llm_output)

    # 提取结构化字段为 JSON（用正则解析）
    steps = []
    pattern = re.compile(
        r"[-–]\s*Target 3D position:\s*\(([^)]+)\)\s*"
        r"[-–]\s*Target orientation:\s*\(([^)]+)\)\s*"
        r"[-–]\s*Gripper action:\s*(.*?)\s*\(?(\d*\.?\d+)?%?\)?\s*"
        r"[-–]\s*Description:\s*(.+)",
        re.IGNORECASE
    )

    matches = pattern.findall(llm_output)

    for i, match in enumerate(matches):
        pos_str, quat_str, action, value, desc = match
        pos = [float(x.strip()) for x in pos_str.split(",")]
        quat = [float(x.strip()) for x in quat_str.split(",")]
        value = float(value) / 100 if value else 0.0
        steps.append({
            "step_index": i + 1,
            "position": pos,
            "quaternion": quat,
            "gripper_action": action.lower(),
            "gripper_value": value,
            "description": desc.strip()
        })

    # 保存 JSON 文件
    with open(save_json_path, "w") as f:
        json.dump(steps, f, indent=2)
    print(f"\n✅ Trajectory steps saved to: {save_json_path}")

    return llm_output, steps
