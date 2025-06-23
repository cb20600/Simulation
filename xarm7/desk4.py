# desk4.py
import os
import genesis as gs
import numpy as np
# os.environ['PYOPENGL_PLATFORM'] = 'glx'
import sys

def create_scene(enable_gui=True):
    print("enable_gui =", enable_gui)
    gs.init(backend=gs.gpu)
    scene = gs.Scene(
        show_viewer = enable_gui,
        viewer_options = gs.options.ViewerOptions(
            res = (1280, 960),
            camera_pos =(1.5, 1.5, 1),
            camera_lookat = (0.0, 0.0, 0.2),
            camera_fov = 45,
            max_FPS = 60,
        ),
        vis_options = gs.options.VisOptions(
            show_world_frame = True, # 显示原点坐标系
            world_frame_size = 1.0, # 坐标系长度(米)
            plane_reflection=False,  # 可选：如果担心反光导致干扰
            # show_cameras = True, 
        ),
            show_FPS=False
    )

    # scene.set_ambient_light((1,1,1), intensity=0.3)
    # scene.add_directional_light(direction=(-1, -1, -1), intensity=3.0)

    table_height = 0.26
    scene.add_entity(gs.morphs.Plane())
    desk=scene.add_entity(gs.morphs.Mesh(
        file="models/components/white_desk/white_desk.glb",
        pos=(0, 0, 0),
        scale = 0.5,
        euler=(90, 0, 0),
        fixed=True,
        collision=True,
        convexify=False,
    ))
    # box = scene.add_entity(gs.morphs.Box(
    #     pos=(0, 0, 0.36),       # 放置位置
    #     size=(0.8, 0.4, 0.02),  # XYZ 方向的长宽高
    #     fixed=True,            # 不参与物理移动
    #     collision=True,        # 有碰撞
    #     visualization=True     # 可视化显示
    # ))

    xarm7 = scene.add_entity(gs.morphs.MJCF(
        file="models/ufactory_xarm7/xarm7.xml",
        collision=True,
        pos=(-0.3, 0.0, table_height)
        ),
        # vis_mode="collision"
        vis_mode="visual"
    )
    # 添加相机
    camera = scene.add_camera(
        # res=(1280, 960),
        # res=(2560, 1440),
        res = (640,640),
        pos=(0, 0.0, 1.0),    # 这里位置是临时的，后面会跟随机械臂末端
        lookat=(0.0, 0.0, 0.0),
        fov=60,
        GUI=enable_gui,
        denoise = True,
        aperture = 3.2,
        # visuliza = True,
    )
    
    # 挂相机到机械臂末端
    link_name = "xarm_gripper_base_link"  # 末端夹爪基座 link
    gripper_link = xarm7.get_link(link_name)

    fruit_x = 0.2
    fruit_y = 0.0
    fruit_z = table_height + 0.14

    banana_x = fruit_x + 0.08
    banana_y = fruit_y + 0.05
    banana_z = fruit_z - 0.01

    carrot_x = fruit_x
    carrot_y = fruit_y - 0.08
    carrot_z = fruit_z - 0.01

    corn_x = fruit_x + 0.08
    corn_y = fruit_y - 0.01
    corn_z = fruit_z

    lemon_x = fruit_x - 0.02
    lemon_y = fruit_y
    lemon_z = fruit_z - 0.01

    lime_x = fruit_x
    lime_y = fruit_y
    lime_z = fruit_z + 0.05

    potato_x = fruit_x - 0.07
    # potato_y = fruit_y - 0.03
    potato_y = fruit_y + 0.07
    potato_z = fruit_z - 0.03
    # potato_z = fruit_z + 0.07

    redpepper_x = fruit_x - 0.05
    redpepper_y = fruit_y - 0.05
    redpepper_z = fruit_z - 0.02

    strawberry_x = fruit_x
    strawberry_y = fruit_y + 0.06
    strawberry_z = fruit_z

    tomato_x = fruit_x
    tomato_y = fruit_y -0.05
    tomato_z = fruit_z

    fruit_fixed = False

    banana = scene.add_entity(gs.morphs.Mesh(
        file="models/components/fruits/banana/banana.glb",
        pos = (banana_x,  banana_y, banana_z),
        euler=(0, 0, 90),
        scale=0.12,
        fixed=fruit_fixed,
        visualization=True,
        collision=True,
    ))

    carrot = scene.add_entity(
        gs.morphs.Mesh(
        file="models/components/fruits/carrot/carrot.glb",
        pos=(carrot_x, carrot_y, carrot_z),
        euler=(0, 90, 40),
        scale=0.006,
        fixed=fruit_fixed,
        visualization=True,
        collision=True,
    ))

    corn = scene.add_entity(gs.morphs.Mesh(
        file="models/components/fruits/corn/corn.glb",
        pos = (corn_x, corn_y, corn_z),
        euler=(0, 0, 30),
        scale=0.0005,
        fixed=fruit_fixed,
        visualization=True,
        collision=True,
    ))   

    lemon = scene.add_entity(gs.morphs.Mesh(
        file="models/components/fruits/lemon/lemon.glb",
        pos = (lemon_x, lemon_y, lemon_z),
        euler=(90, 90, 0),
        scale=0.004,
        fixed=fruit_fixed,
        visualization=True,
        collision=True,
    ))

    # lime = scene.add_entity(gs.morphs.Mesh(
    #     file="models/components/fruits/lime/lime.glb",
    #     pos = (lime_x, lime_y, lime_z),
    #     euler=(0, 0, 0),
    #     scale=0.002,
    #     fixed=fruit_fixed,
    #     visualization=True,
    #     collision=True,
    #     convexify=True,
    #     merge_submeshes_for_collision=True,
    # ))    

    # # box = scene.add_entity(gs.morphs.Box(
    # #     pos=(0.2, 0.0, 0.4),       # 放置位置
    # #     size=(0.04, 0.03, 0.03),
    # #     fixed=False,  
    # #     collision=True,
    # #     visualization=True 
    # # ))

    potato = scene.add_entity(gs.morphs.Mesh(
        file="models/components/fruits/potato/potato.glb",
        pos = (potato_x, potato_y, potato_z),
        euler=(0, 0, 60),
        scale=0.023,
        fixed=fruit_fixed,
        visualization=True,
        collision=True,
    ))

    redpepper = scene.add_entity(gs.morphs.Mesh(
        file="models/components/fruits/redpepper/redpepper.glb",
        pos = (redpepper_x, redpepper_y, redpepper_z),
        euler=(90, 0, 60),
        scale=0.0008,
        fixed=fruit_fixed,
        visualization=True,
        collision=True,
    ))

    strawberry = scene.add_entity(gs.morphs.Mesh(
        file="models/components/fruits/strawberry/strawberry1.glb",
        pos = (strawberry_x, strawberry_y, strawberry_z),
        euler=(0, 0, 60),
        scale=0.07,
        fixed=fruit_fixed,
        visualization=True,
        collision=True,
    ))
    '''
    # tomato = scene.add_entity(gs.morphs.Mesh(
    #     file="models/components/fruits/tomato/Tomato.obj",
    #     pos = (tomato_x, tomato_y, tomato_z),
    #     euler=(90, 0, 60),
    #     scale=0.025,
    #     fixed=True,
    #     visualization=True,
    #     collision=True,
    # ))    

    # yellowBin =  scene.add_entity(
    #     gs.morphs.Mesh(
    #         file="models/components/trashbin/Trashbin.glb",
    #         pos=(0.55, 0.1, 0.05),
    #         euler=(90, 0, 0),
    #         scale=0.02,
    #         visualization=True,
    #         fixed=True,
    #         collision=False,
    #         convexify=False
    #     ))

    # redBin =  scene.add_entity(
    #     gs.morphs.Mesh(
    #         file="models/components/trashbin/Trashbin.glb",
    #         pos=(0.55, -0.1, 0.05),
    #         euler=(90, 0, 0),
    #         scale=0.02,
    #         visualization=True,
    #         fixed=True,
    #         collision=False,
    #         convexify=False
    #     ))

    # light_plane = scene.add_entity(gs.morphs.Box(
    #     size=(0.05,0.05, 0.02), 
    #     pos=(0.25, 0.0, 1.5),     # 水果上方的某个高度
    #     euler=(180, 0, 0),          # 朝下（绕X轴旋转180度）
    #     # scale=(0.1, 0.1, 0.01),   # 一个小灯面板
    #     # color=(0.1, 0.1, 0.1, 0.0),  # 几乎不可见
    #     fixed=True,
    # ))

    # # 添加光源绑定到该面板上
    # scene.add_light(
    #     morph=light_plane,
    #     color=(1.0, 1.0, 1.0, 1.0),  # 白色光
    #     intensity=50.0,             # 可调（建议10-50之间）
    #     revert_dir=True,           # 默认朝表面法线方向发光
    #     double_sided=False,
    #     beam_angle=180.0            # 广角柔光
    # )

    # scene.add_light(
    #     morph=camera,
    #     color=(1.0, 1.0, 1.0, 1.0), 
    #     intensity=20.0, 
    #     revert_dir=False, 
    #     double_sided=False, 
    #     beam_angle=180.0
    #     )
    '''
    scene.build()

    T = np.array([
        [1,  0,  0,  -0.08],
        [0, -1,  0,  0],
        [0,  0, -1,  0.1],
        [0,  0,  0,  1]
    ])
    '''
    # T = np.array([
    #     [ 0, -1,  0, -0.08],  # X': -Y
    #     [ 1,  0,  0,  0   ],  # Y':  X
    #     [ 0,  0, -1,  0.05],  # ✅ Z': -Z ✅ 向下看 ✅ 光照正常
    #     [ 0,  0,  0,  1   ]
    # ])
    '''
    camera.attach(rigid_link=gripper_link, offset_T=T)

    fruit_entities = {
        "banana": banana,
        "carrot": carrot,
        "corn": corn,
        "lemon": lemon,
        # "lime": lime,
        "potato": potato,
        "redpepper":redpepper,
        "strawberry":strawberry,

    }
    # for i, (name, fruit) in enumerate(fruit_entities.items()):
    #     print(f"水果 {i} 的 name 是：{name}, 实体类型: {type(fruit)}")

    # fruit_entities = [
    #     banana,
    #     carrot,
    #     corn,
    #     lemon,
    #     lime,
    #     potato,
    #     redpepper,
    #     strawberry
    # ]

    bins = {
        # "redBin": redBin,
        # "yellowBin": yellowBin,
    }

    return scene, xarm7, fruit_entities, bins, camera

# if __name__ == "__main__":
    # scene, xarm7, fruits, bins, camera = create_scene()

    # for i in range(5000):
    #     scene.step()
    #     camera.render(
    #         rgb = True,
    #         # depth = True,
    #     )
    # print("✅ 仿真完成")

########################## main ##########################
if __name__ == "__main__":
    enable_gui=True
    scene, xarm7, fruits, bins, camera = create_scene(enable_gui)
    print("等待物理系统稳定...")

    for _ in range(100):
        scene.step()

    # lime = fruits["lime"]
    # print("lime:", lime.get_pos())
    jnt_names = [
        "joint1", "joint2", "joint3",
        "joint4", "joint5", "joint6", "joint7"
    ]
    dofs_idx = [xarm7.get_joint(name).dof_idx_local for name in jnt_names]

    # 2. PID & 力矩限制设置
    xarm7.set_dofs_kp(
        kp=np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000]),
        dofs_idx_local=dofs_idx
    )
    xarm7.set_dofs_kv(
        kv=np.array([450, 450, 350, 350, 200, 200, 200]),
        dofs_idx_local=dofs_idx
    )
    xarm7.set_dofs_force_range(
        lower=np.array([-87, -87, -87, -87, -12, -12, -12]),
        upper=np.array([ 87,  87,  87,  87,  12,  12,  12]),
        dofs_idx_local=dofs_idx
    )

    # 3. 定义初始和目标关节角
    pos_init   = np.array([0, 0, 0, 1, 0, 0.5, 0])
    for _ in range(100):
        scene.step()

    # 6. 重置到初始姿态
    for i in range(100):
        xarm7.set_dofs_position(pos_init, dofs_idx)
        scene.step()

    # if enable_gui==True:
    #     camera.render(rgb=True)

    # 设置末端执行器为夹爪基座
    end_effector = xarm7.get_link("xarm_gripper_base_link")
    # 设置显示该 link 的坐标轴

    end_pos = end_effector.get_pos().cpu().numpy()
    print("末端执行器位置：",end_pos)
    
    base_pos = xarm7.get_link("xarm_gripper_base_link").get_pos()
    left_finger_pos = xarm7.get_link("left_finger").get_pos()
    right_finger_pos = xarm7.get_link("right_finger").get_pos()

    print("gripper_base_pos     = [{:.3f}, {:.3f}, {:.3f}]".format(*base_pos))
    print("left_finger_pos      = [{:.3f}, {:.3f}, {:.3f}]".format(*left_finger_pos))
    print("right_finger_pos     = [{:.3f}, {:.3f}, {:.3f}]".format(*right_finger_pos))

    grasp_point = (left_finger_pos + right_finger_pos) / 2
    gripper_offset = grasp_point - base_pos

    print("⚙️ gripper_offset     = [{:.3f}, {:.3f}, {:.3f}]".format(*gripper_offset))



    # 设置目标位置和方向（竖直向下）
    target_pos = np.array([0.20, 0.0, 1])
    target_quat = np.array([0, 0, 1, 0])  # Z轴朝下，姿态可微调

    # 使用逆运动学求解目标姿态的关节位置
    qpos = xarm7.inverse_kinematics(
        link=end_effector,
        pos=target_pos,
        quat=target_quat,
    )

    # 设置夹爪张开（假设末6个自由度是夹爪）
    qpos[7:] = 0.04

    # 生成运动路径
    path = xarm7.plan_path(qpos_goal=qpos, num_waypoints=200)

    # 执行路径
    if enable_gui:
        for waypoint in path:
            xarm7.control_dofs_position(waypoint)
            scene.step()
            camera.render(rgb=True)
    else:
        for waypoint in path:
            xarm7.control_dofs_position(waypoint)
            scene.step()

    # control_xarm7(scene, xarm7, cam_attachment, camera)
    # print(camera.transform)
    for i in range(5000):
        scene.step()
    print("✅ 控制结束")