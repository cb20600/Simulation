import numpy as np

class WeldConstraintSim:
    def __init__(self, robot_entity, left_link_name, right_link_name, cube_entity, offset=None):
        self.robot = robot_entity
        self.cube = cube_entity
        self.left_link = robot_entity.get_link(name=left_link_name)
        self.right_link = robot_entity.get_link(name=right_link_name)
        self.left_idx = self.left_link.idx_local
        self.right_idx = self.right_link.idx_local
        self.offset = offset if offset is not None else np.array([0.0, 0.0, -0.03])
        self.active = True

    def step(self):
        if not self.active:
            return
        pos_tensor = self.robot.get_links_pos()
        quat_tensor = self.robot.get_links_quat()
        pos_l = pos_tensor[self.left_idx].cpu().numpy()
        pos_r = pos_tensor[self.right_idx].cpu().numpy()
        center_pos = (pos_l + pos_r) / 2 + self.offset
        self.cube.set_pos(center_pos)

    def remove(self):
        self.active = False
        self.cube = None  # 解除引用，避免误调用 set_pos()

