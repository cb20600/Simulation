import os
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from desk4 import create_scene  # 你需要把 create_scene 函数所在的文件名换成实际文件名

SAVE_DIR = "captured_images"
os.makedirs(SAVE_DIR, exist_ok=True)

def set_initial_pose(xarm7, scene):
    pos_init = np.array([0, 0, 0, 1, 0, 0.8, 0])
    dofs_idx = list(range(7))
    xarm7.control_dofs_position(pos_init, dofs_idx)
    for _ in range(100):
        scene.step()

def capture_view(scene, arm, camera, end_link, target_pos, target_quat, img_id):
    try:
        qpos = arm.inverse_kinematics(link=end_link, pos=target_pos, quat=target_quat)
    except Exception as e:
        print(f"❌ IK Failed at {target_pos}, error: {e}")
        return

    path = arm.plan_path(qpos_goal=qpos, num_waypoints=100)
    for wp in path:
        arm.control_dofs_position(wp)
        scene.step()

    for _ in range(10):
        scene.step()

    img = camera.render(rgb=True)
    if isinstance(img, dict) and "rgb" in img:
        out_path = os.path.join(SAVE_DIR, f"img_{img_id:03d}.png")
        cv2.imwrite(out_path, img["rgb"][..., ::-1])
        print(f"✅ Saved: {out_path}")

def generate_views(center, radii=[0.35], heights=[0.45, 0.5], angle_step=30):
    views = []
    for h in heights:
        for r in radii:
            for angle in range(0, 360, angle_step):
                rad = np.radians(angle)
                x = center[0] + r * np.cos(rad)
                y = center[1] + r * np.sin(rad)
                z = h
                pos = np.array([x, y, z])

                forward = center - pos
                forward /= np.linalg.norm(forward)
                up = np.array([0, 0, 1])
                right = np.cross(up, forward)
                right /= np.linalg.norm(right)
                up = np.cross(forward, right)
                rot = np.stack([right, up, forward], axis=1)
                quat_xyzw = R.from_matrix(rot).as_quat()
                quat_wxyz = np.roll(quat_xyzw, 1)
                views.append((pos, quat_wxyz))
    return views

if __name__ == "__main__":

    scene, xarm7, fruits, bins, camera = create_scene()
    ee_link = xarm7.get_link("xarm_gripper_base_link")
    fruit_center = np.array([0.3, 0.0, 0.4])

    set_initial_pose(xarm7, scene)

    views = generate_views(center=fruit_center)
    for idx, (pos, quat) in enumerate(views):
        capture_view(scene, xarm7, camera, ee_link, pos, quat, idx)

    print("✅ 图像采集完成")
