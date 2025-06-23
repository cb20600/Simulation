import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import genesis as gs
import numpy as np
import time 

class MyCobotROS2(Node):
    def __init__(self):
        super().__init__('genesis_mycobot_controller')

        # 订阅 ROS 2 关节角度话题
        self.subscription = self.create_subscription(
            Float64MultiArray,
            '/mycobot/joint_positions',
            self.ros2_callback,
            10
        )

        # 初始化 Genesis
        gs.init(backend=gs.gpu)
        self.get_logger().info("Genesis 初始化完成")

        # 创建仿真场景
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.5, 3, 1),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                max_FPS=60,
            ),
            sim_options=gs.options.SimOptions(dt=0.01),
            show_viewer=True,
        )

        # 添加地面
        self.plane = self.scene.add_entity(gs.morphs.Plane())

        # 添加机械臂
        self.mycobot = self.scene.add_entity(
            gs.morphs.MJCF(file="/home/ubuntu1/models/mujoco_mycobot-main/mujoco_mycobot-main/mycobot_with_gripper.xml"),
        )

        # 获取关节信息
        self.motors_dof = np.arange(6)

        # 绑定仿真循环
        self.scene.on_step = self.simulation_step

        # 运行仿真
        self.scene.build()
        self.get_logger().info("Genesis 机械臂仿真启动")

    def ros2_callback(self, msg):
        """ 处理 ROS 2 关节控制数据，并进行插值 """
        if len(msg.data) != 6:
            self.get_logger().warn("接收到的关节数不匹配！应为6个")
            return

        # ✅ 这里使用局部变量，而不是 `self.target_qpos`
        target_qpos = np.array(msg.data[:6])
        self.get_logger().info(f"接收到 ROS 2 指令: {target_qpos}")

        # ✅ 获取当前机械臂的关节角度，并转换到 NumPy 格式
        current_qpos = self.mycobot.get_dofs_position().cpu().numpy()[:6]

        # 生成平滑运动轨迹（10 个插值点）
        num_steps = 10
        for i in range(num_steps):
            intermediate_qpos = current_qpos + (target_qpos - current_qpos) * (i / num_steps)
            self.mycobot.control_dofs_position(intermediate_qpos, self.motors_dof)
            self.scene.step()  # 让仿真继续

        # 确保最终到达目标
        self.mycobot.control_dofs_position(target_qpos, self.motors_dof)
        self.scene.step()

    def simulation_step(self):
        """ 每个仿真步执行的操作 """
        if hasattr(self, 'target_qpos'):
            self.mycobot.control_dofs_position(self.target_qpos, self.motors_dof)


if __name__ == '__main__':
    rclpy.init()
    node = MyCobotROS2()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


# ros2 topic pub /mycobot/joint_positions std_msgs/msg/Float64MultiArray "{data: [0.0, -0.5, 0.5, -0.5, 0.5, 0.0]}"