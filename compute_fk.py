
import numpy as np
import torch
import random
import os
from differentiable_robot_model.robot_model import (
    DifferentiableRobotModel
)


torch.set_printoptions(precision=3, sci_mode=False)
random.seed(0)
np.random.seed(1)
torch.manual_seed(0)

######### SOLVED!
class DifferentiableFrankaPanda(DifferentiableRobotModel):
    def __init__(self, device=None):
        #rel_urdf_path = "panda_description/urdf/panda_no_gripper.urdf"
        #self.urdf_path = os.path.join(robot_description_folder, rel_urdf_path)
        self.urdf_path = '/home/crslab/GRASP/differentiable-robot-model/diff_robot_data/panda_description/urdf/panda_hand_arm_modified_tas.urdf'
        self.learnable_rigid_body_config = None
        self.name = "differentiable_franka_panda"
        super().__init__(self.urdf_path, self.name, device=device)

# device="cpu"
device = torch.device('cuda:0')

gt_robot_model = DifferentiableFrankaPanda(device=device)

q = torch.zeros([1,9], dtype=torch.float32).to(device)#
q = torch.FloatTensor([-2.17801821, -1.21684093,  1.69959079, -1.37832062,  0.79249172,  1.16310331, -1.74444444, 0, 0]).to(device)
a, b = gt_robot_model.compute_forward_kinematics(q=q, link_name="panda_gripper_center")

print(a)
print(b)
