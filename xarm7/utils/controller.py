import os
import json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

import torch
os.environ['PYOPENGL_PLATFORM'] = 'glx'

import genesis as gs
from desk4 import create_scene
from gripper_utils import init_gripper_controller, close_gripper, open_gripper, move_gripper_to

# ========== Step 1: 初始化 OpenAI ==========
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chat_with_gpt(messages):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.4
    )
    return response.choices[0].message

def clean_json_output(text):
    import re
    text = text.strip()
    if text.startswith("```json") or text.startswith("```"):
        text = re.sub(r"```(?:json)?", "", text)
        text = text.strip("`\n")
    return text

# ========== Step 2: 执行模块函数 ==========
def perform_pick_and_place(scene, xarm7, end_effector, obj_pos, target_pos, quat):
    motors_dof = np.arange(7)
    fingers_dof = np.arange(11, 13)
    offset = np.array([0.053, 0.0, 0.15])

    pos1 = obj_pos + offset + np.array([0.0, 0.0, 0.18])
    qpos = xarm7.inverse_kinematics(link=end_effector, pos=pos1, quat=quat)
    qpos[7:] = 0.02
    path = xarm7.plan_path(qpos_goal=qpos, num_waypoints=100)
    for waypoint in path:
        xarm7.control_dofs_position(waypoint)
        scene.step()

    reach_pos = pos1 + np.array([0.0, 0.0, -0.15])
    qpos = xarm7.inverse_kinematics(link=end_effector, pos=reach_pos, quat=quat)
    path = xarm7.plan_path(qpos_goal=qpos, num_waypoints=100)
    for waypoint in path:
        xarm7.control_dofs_position(waypoint[:7], motors_dof)
        scene.step()

    qpos[7:] = 0.4
    xarm7.control_dofs_position(qpos)
    xarm7.control_dofs_force(np.array([-1.5, -1.5]), fingers_dof)
    for _ in range(100): scene.step()

    pos_lift = reach_pos + np.array([0.0, 0.0, 0.15])
    qpos = xarm7.inverse_kinematics(link=end_effector, pos=pos_lift, quat=quat)
    xarm7.control_dofs_position(qpos[:7], motors_dof)
    for _ in range(100): scene.step()

    move_pos = target_pos + np.array([0.0, 0.0, 0.2])
    qpos = xarm7.inverse_kinematics(link=end_effector, pos=move_pos, quat=quat)
    path = xarm7.plan_path(qpos_goal=qpos, num_waypoints=100)
    for waypoint in path:
        xarm7.control_dofs_position(waypoint[:7], motors_dof)
        scene.step()

    end_pos = target_pos + np.array([0.0, 0.0, 0.0])
    qpos = xarm7.inverse_kinematics(link=end_effector, pos=end_pos, quat=quat)
    path = xarm7.plan_path(qpos_goal=qpos, num_waypoints=100)
    for waypoint in path:
        xarm7.control_dofs_position(waypoint[:7], motors_dof)
        scene.step()

    qpos[7:] = 0.0
    xarm7.control_dofs_position(qpos)
    xarm7.control_dofs_force(np.array([-5, -5]), fingers_dof)
    for _ in range(100): scene.step()

# ========== Step 3: 子任务执行控制器 ==========
def execute_subtasks(subtasks, scene, xarm7, fruits, bins, end_effector):
    quat = np.array([0, 0, 1, 0])
    for subtask in subtasks:
        obj_name = subtask.get("object_name")
        target_name = subtask.get("target_bin")
        ref_object = subtask.get("ref_object")
        direction = subtask.get("direction")
        action = subtask.get("action")

        if obj_name not in fruits:
            print(f"❌ 无效的对象名: {obj_name}")
            continue

        if action == "move" and ref_object and direction:
            if ref_object not in fruits:
                print(f"❌ 无效的参考对象: {ref_object}")
                continue
            # 简单处理方向为 right side，向 x 轴正方向偏移
            offset = np.array([0.12, 0, 0]) if "right" in direction else np.array([-0.12, 0, 0])
            target_pos = fruits[ref_object].get_pos().cpu().numpy() + offset
        elif action == "move" and target_name in bins:
            target_pos = bins[target_name].get_pos().cpu().numpy()
        else:
            print(f"❌ 无法解析目标位置: {target_name or direction}")
            continue

        obj_pos = fruits[obj_name].get_pos().cpu().numpy()
        print(f"🚚 移动 {obj_name} 到 {direction or target_name}")
        perform_pick_and_place(scene, xarm7, end_effector, obj_pos, target_pos, quat)

# ========== Step 4: 主函数入口 ==========
if __name__ == "__main__":
    print("💬 启动智能机械臂控制系统 (输入 q 退出)")

    enable_gui = True
    scene, xarm7, fruits, bins, camera = create_scene(enable_gui)
    for _ in range(100): scene.step()
    end_effector = xarm7.get_link("xarm_gripper_base_link")

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ["q", "quit", "exit"]:
            print("👋 退出系统")
            break

        stage1_messages = [{
            "role": "system",
            "content": (
                "You are an expert at interpreting robot control instructions.\n"
                "The user will give you a natural language command, and your job is to extract a structured list of triplets.\n"
                "Each item should contain:\n"
                "- object: the object to manipulate\n"
                "- action: move / rotate / align / place_near\n"
                "- target: bin name, reference object, or direction info\n"
                "Output a JSON array of triplets."
            )
        }, {
            "role": "user",
            "content": user_input
        }]

        stage1_result = chat_with_gpt(stage1_messages)
        try:
            triplets_raw = clean_json_output(stage1_result.content)
            triplets = json.loads(triplets_raw)
        except Exception as e:
            print("⚠️ 无法解析 triplets:", e)
            print("原始输出:", stage1_result.content)
            continue

        stage2_messages = [{
            "role": "system",
            "content": (
                "You are a robot task planner.\n"
                "Given a list of structured object-action-target triplets, generate a list of subtasks.\n"
                "If the 'target' refers to a relative direction (e.g., 'right side of banana'), extract the direction and reference object separately.\n"
                "Each subtask should include:\n"
                "- task_id: index starting from 1\n"
                "- description: natural language description\n"
                "- action: same as input (move, align, etc.)\n"
                "- object_name: name of the object to move\n"
                "- target_bin: if the target is a bin, use its name\n"
                "- ref_object: if the target is relative (like 'right side of banana'), extract 'banana'\n"
                "- direction: for relative placement (like 'right side')\n"
                "Output a JSON array of subtasks ready for planning."
            )
        }, {
            "role": "user",
            "content": json.dumps(triplets, ensure_ascii=False)
        }]

        stage2_result = chat_with_gpt(stage2_messages)
        try:
            subtasks_raw = clean_json_output(stage2_result.content)
            subtasks = json.loads(subtasks_raw)
            print(json.dumps(subtasks, indent=2, ensure_ascii=False))
        except Exception as e:
            print("⚠️ 无法解析 subtasks:", e)
            print("原始输出:", stage2_result.content)
            continue

        execute_subtasks(subtasks, scene, xarm7, fruits, bins, end_effector)
