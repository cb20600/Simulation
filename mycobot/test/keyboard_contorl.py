import numpy as np
import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'
import genesis as gs

# 初始化 Genesis
gs.init(backend=gs.gpu)

# 创建场景并启用自定义键盘控制
scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(0, -1.5, 1.5),
        camera_lookat=(0, 0, 0.5),
        camera_fov=45,
        max_FPS=60,
        # 启用自定义键盘回调
        enable_custom_keyboard=True,  # 假设 Genesis 支持此选项
    ),
    sim_options=gs.options.SimOptions(dt=0.01),
    show_viewer=True,
)

# 加载机械臂模型
plane = scene.add_entity(gs.morphs.Plane())
mycobot = scene.add_entity(gs.morphs.MJCF(file="/home/ubuntu1/models/mujoco_mycobot-main/mujoco_mycobot-main/mycobot_with_gripper.xml"))
scene.build()

# ================== 定义键盘回调函数 ==================
def keyboard_callback(key, action, mods):
    global target_pos
    step_arm = 0.1
    step_gripper = 0.02

    # 只处理按下事件（action=1 表示按下）
    if action == 1:
        # 机械臂关节控制
        if key == gs.KEY_Q:
            target_pos[0] += step_arm
        elif key == gs.KEY_A:
            target_pos[0] -= step_arm
        elif key == gs.KEY_W:
            target_pos[1] += step_arm
        elif key == gs.KEY_S:
            target_pos[1] -= step_arm
        elif key == gs.KEY_E:
            target_pos[2] += step_arm
        elif key == gs.KEY_D:
            target_pos[2] -= step_arm
        elif key == gs.KEY_R:
            target_pos[3] += step_arm
        elif key == gs.KEY_F:
            target_pos[3] -= step_arm
        elif key == gs.KEY_T:
            target_pos[4] += step_arm
        elif key == gs.KEY_G:
            target_pos[4] -= step_arm
        elif key == gs.KEY_Y:
            target_pos[5] += step_arm
        elif key == gs.KEY_H:
            target_pos[5] -= step_arm
        # 夹爪控制（空格键）
        elif key == gs.KEY_SPACE:
            target_pos[6:] += step_gripper
            target_pos[6:] = np.clip(target_pos[6:], 0, 0.04)

# 注册键盘回调到场景
scene.viewer.set_keyboard_callback(keyboard_callback)