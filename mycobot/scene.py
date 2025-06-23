import os
import genesis as gs

os.environ['PYOPENGL_PLATFORM'] = 'glx'
gs.init(backend=gs.gpu)

def create_scene():
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, 1.5, 1),
            camera_lookat=(0.0, 0.0, 0.2),
            camera_fov=45,
        ),
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=True,
    )

    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(gs.morphs.Plane(        
        pos=(-1.5, 0, 0),
        euler=(0, 90, 0),
    ))
    scene.add_entity(gs.morphs.Plane(        
        pos=(0, -2, 0),
        euler=(90, 0, 0),
    ))

    # 添加桌椅
    scene.add_entity(gs.morphs.Mesh(
        file="models/components/dining_table_chair.glb",
        pos=(0, 0, 0),
        euler=(90, 0, 0),
        scale=0.5,
        fixed=True,
        collision=True,
        convexify=False,
    ))

    # 添加机械臂
    mycobot = scene.add_entity(gs.morphs.MJCF(
        file="models/mujoco_mycobot-main/mujoco_mycobot-main/mycobot_with_gripper1.xml",
        euler=(0, 0, 30),
        pos=(0.28, 0.0, 0.38)
    ))

    # 添加水果
    apple = scene.add_entity(gs.morphs.Mesh(
        file="models/components/fruits/apple/10162_Apple_v01_l3.obj",
        pos=(0.45, -0.1, 0.375),
        scale=0.00019,
        fixed=False,
        visualization=True,
        collision=True,
    ))
    pear = scene.add_entity(gs.morphs.Mesh(
        file="models/components/fruits/pear.glb",
        pos=(0.12, -0.15, 0.385),
        scale=0.003,
        fixed=False,
        visualization=True,
        collision=True,
    ))
    banana = scene.add_entity(gs.morphs.Mesh(
        file="models/components/fruits/banana.glb",
        pos=(0.43, 0, 0.39),
        euler=(90, 0, 60),
        scale=0.12,
        fixed=False,
        visualization=True,
        collision=True,
    ))
    orange = scene.add_entity(gs.morphs.Mesh(
        file="models/components/fruits/12204_Fruit_v1_L3.obj",
        pos=(0.35, 0.15, 0.375),
        scale=0.003,
        fixed=False,
        visualization=True,
        collision=True,
    ))

    yellowBin =  scene.add_entity(
        gs.morphs.Mesh(
            file="models/components/trashbin/Trashbin.glb",
            pos=(0.55, 0.1, 0.05),
            euler=(90, 0, 0),
            scale=0.02,
            visualization=True,
            fixed=True,
            collision=False,
            convexify=False
        ))

    redBin =  scene.add_entity(
        gs.morphs.Mesh(
            file="models/components/trashbin/Trashbin.glb",
            pos=(0.55, -0.1, 0.05),
            euler=(90, 0, 0),
            scale=0.02,
            visualization=True,
            fixed=True,
            collision=False,
            convexify=False
        ))


    scene.build()

    fruit_entities = {
        "apple": apple,
        "banana": banana,
        "orange": orange,
        "pear": pear,
    }

    bins = {
        "redBin": redBin,
        "yellowBin": yellowBin,
    }
    
    return scene, mycobot, fruit_entities, bins

