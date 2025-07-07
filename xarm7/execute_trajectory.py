import json
import numpy as np
from numpy.linalg import norm

def clamp(val, min_val, max_val):
    return max(min_val, min(max_val, val))

def is_valid_pose(pos, quat):
    if len(pos) != 3 or len(quat) != 4:
        return False
    if not all(isinstance(x, (int, float)) for x in pos + quat):
        return False
    if np.any(np.isnan(pos)) or np.any(np.isnan(quat)):
        return False
    return True

def execute_trajectory(xarm7, gripper, trajectory_json_path, scene=None, wait_steps=30):
    """
    安全执行 JSON 中的轨迹指令，控制机械臂移动并控制夹爪动作。

    参数:
        xarm7: 机械臂对象
        gripper: GripperController 实例
        trajectory_json_path: 轨迹 JSON 路径
        scene: 可选，Genesis 场景对象用于 step()
        wait_steps: 每步等待的 step 次数
    """
    try:
        with open(trajectory_json_path, "r") as f:
            steps = json.load(f)
    except Exception as e:
        print(f"❌ 无法读取轨迹文件: {e}")
        return

    if not isinstance(steps, list) or len(steps) == 0:
        print("❌ JSON 轨迹格式错误或为空，无法执行。")
        return

    print(f"✅ 开始执行轨迹，共 {len(steps)} 步")

    for step in steps:
        try:
            idx = step.get("step_index", "?")
            pos = step["position"]
            quat = step["quaternion"]
            action = step.get("gripper_action", "maintain grip")
            value = clamp(step.get("gripper_value", 0.0), 0.0, 1.0)
            desc = step.get("description", "")

            print(f"\n🔧 Step {idx}: {desc}")
            print(f"  ▶️ Pos: {pos}, Quat: {quat}, Gripper: {action} ({value:.2f})")

            # 校验姿态数据
            if not is_valid_pose(pos, quat):
                print("⚠️ 无效的 position 或 quaternion，跳过此步")
                continue

            # 限制 Z 值高度，防止撞桌面
            if not (0.1 < pos[2] < 1.5):
                print(f"⚠️ Z 值异常：{pos[2]}，跳过此步")
                continue

            # 控制机械臂
            xarm7.set_pose(pos, quat)

            # 控制夹爪
            if action == "open":
                gripper.open()
            elif action == "close":
                gripper.close(value)
            elif action == "maintain grip":
                pass
            else:
                print(f"⚠️ 未知夹爪动作类型：{action}")

            # 步进仿真
            if scene:
                for _ in range(wait_steps):
                    scene.step()

        except Exception as e:
            print(f"❌ 第 {step.get('step_index', '?')} 步执行失败: {e}")
