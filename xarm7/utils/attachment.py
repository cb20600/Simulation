import numpy as np

class Attachment:
    def __init__(self, parent_link, child_obj, offset_transform=None):
        self.parent_link = parent_link
        self.child_obj = child_obj
        if offset_transform is None:
            self.offset_transform = np.eye(4)
        else:
            self.offset_transform = offset_transform

    def quat_to_rot_matrix(self, q):
        x, y, z, w = q
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        rot = np.array([
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]
        ])
        return rot

    def get_pose_from_link(self, link):
        pos = link.get_pos().cpu().numpy()
        quat = link.get_quat().cpu().numpy()
        rot = self.quat_to_rot_matrix(quat)

        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3] = pos
        return pose

    def update(self):
        parent_pose = self.get_pose_from_link(self.parent_link)
        child_pose = parent_pose @ self.offset_transform
        self.child_obj.set_pose(transform=child_pose)
