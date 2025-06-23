import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'

import genesis as gs
import numpy as np
from scene import create_scene
from constraints_sim import WeldConstraintSim
from llm_controller import run_llm_command
from dotenv import load_dotenv
import os
from openai import OpenAI
from llm_controller import run_llm_command

# DeepSeek API
load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

# Init scene and entities
scene, mycobot, fruits, bins = create_scene()

# Joint
motors_dof = np.arange(6)

gripper_left_dofs = [
    mycobot.get_joint("gripper_left2_to_gripper_left1").dof_idx_local,
    mycobot.get_joint("gripper_base_to_gripper_left2").dof_idx_local,
    mycobot.get_joint("gripper_base_to_gripper_left3").dof_idx_local,
]
gripper_right_dofs = [
    mycobot.get_joint("gripper_right2_to_gripper_right1").dof_idx_local,
    mycobot.get_joint("gripper_base_to_gripper_right2").dof_idx_local,
    mycobot.get_joint("gripper_base_to_gripper_right3").dof_idx_local,
]

gripper_opening = np.array([0.05] * 3)
end_effector = mycobot.get_link("gripper_base")

current_weld = None

def grasp(obj_entity):
    global current_weld

    obj_pos = np.array(obj_entity.get_pos().tolist())
    offset = np.array([0, -0.02, 0])
    quat = np.array([0, 1, 0, 0])

    qpos = mycobot.inverse_kinematics(
        link=end_effector,
        pos=obj_pos + offset + np.array([0.0, 0.0, 0.15]),
        quat=quat
    )
    path = mycobot.plan_path(qpos_goal=qpos, num_waypoints=100)
    for waypoint in path:
        mycobot.control_dofs_position(waypoint[:-6], motors_dof)
        scene.step()

    qpos = mycobot.inverse_kinematics(
        link=end_effector,
        pos=obj_pos + offset + np.array([0.0, 0.0, 0.1]),
        quat=quat
    )
    path = mycobot.plan_path(qpos_goal=qpos, num_waypoints=60)
    for waypoint in path:
        mycobot.control_dofs_position(waypoint[:-6], motors_dof)
        scene.step()
    for _ in range(40):
        scene.step()

    mycobot.control_dofs_position(-gripper_opening, gripper_left_dofs)
    mycobot.control_dofs_position(gripper_opening, gripper_right_dofs)
    for _ in range(50):
        scene.step()

    obj_entity.fixed = False
    current_weld = WeldConstraintSim(
        robot_entity=mycobot,
        left_link_name="gripper_left3",
        right_link_name="gripper_right3",
        cube_entity=obj_entity,
        offset=np.array([0.0, 0.0, -0.07])
    )
    current_weld.step()
    for _ in range(10):
        current_weld.step()
        scene.step()

    qpos = mycobot.inverse_kinematics(
        link=end_effector,
        pos=obj_pos + offset + np.array([0.0, 0.0, 0.2]),
        quat=quat
    )
    path = mycobot.plan_path(qpos_goal=qpos, num_waypoints=60)
    for waypoint in path:
        mycobot.control_dofs_position(waypoint[:-6], motors_dof)
        current_weld.step()
        scene.step()

def place(drop_pos):
    global current_weld

    offset = np.array([0, -0.02, 0])
    quat = np.array([0, 1, 0, 0])

    qpos = mycobot.inverse_kinematics(
        link=end_effector,
        pos=drop_pos + offset + np.array([0, 0, 0.5]),
        quat=quat
    )
    path = mycobot.plan_path(qpos_goal=qpos, num_waypoints=100)
    for waypoint in path:
        mycobot.control_dofs_position(waypoint[:-6], motors_dof)
        if current_weld:
            current_weld.step()
        scene.step()

    qpos = mycobot.inverse_kinematics(
        link=end_effector,
        pos=drop_pos + offset + np.array([0, 0, 0.45]),
        quat=quat
    )
    path = mycobot.plan_path(qpos_goal=qpos, num_waypoints=60)
    for waypoint in path:
        mycobot.control_dofs_position(waypoint[:-6], motors_dof)
        if current_weld:
            current_weld.step()
        scene.step()

    mycobot.control_dofs_position(gripper_opening, gripper_left_dofs)
    mycobot.control_dofs_position(-gripper_opening, gripper_right_dofs)
    
    for _ in range(50):
        scene.step()

    if current_weld:
        current_weld.remove()
        current_weld = None

# Get object position
apple = fruits["apple"]
banana = fruits["banana"]
redBin = bins["redBin"]
yellowBin = bins["yellowBin"]

red_bin_pos = np.array(redBin.get_pos().tolist())
yellow_bin_pos = np.array(yellowBin.get_pos().tolist())

# Execute
# grasp(apple)
# place(red_bin_pos)
# print("apple finish")
# grasp(banana)
# place(yellow_bin_pos)
# while True:
#     scene.step()

while True:
    cmd = input("\nEnter command (or 'quit'): ")
    if cmd.strip().lower() in ["quit", "exit"]:
        break

    result = run_llm_command(cmd, client, fruits, bins, grasp, place)

    if result == "incomplete":
        incomplete_last = True
    else:
        incomplete_last = None