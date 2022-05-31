import numpy as np
import torch
import random
import os
from differentiable_robot_model.robot_model import (
    DifferentiableRobotModel
)
# from differentiable_robot_model.robot_model import (
#     DifferentiableFrankaPanda
# )




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

device="cpu"
gt_robot_model = DifferentiableFrankaPanda(device=device)
#print(learnable_robot_model.print_learnable_params())
print(gt_robot_model._n_dofs)
print(gt_robot_model._name_to_idx_map)

# q = torch.FloatTensor([[0, 0, 0, 0, 0, 0, 0, 0, 0]])
# q = torch.FloatTensor([[-0.000221715449139051, -0.7847544138724399, 0.00010503894092899076, -2.3551128582357386, -4.516856144455796e-05, 1.5710171426932014, 0.784951122533256, 0, 0.05]])
# q = torch.FloatTensor([[-0.41615460004618293, 0.2536675692106548, -0.11877543107982268, -1.281291567250302, 1.0011549899256862, 2.3353868778729256, 0.6946935629604591, 0, 0]])
q = torch.FloatTensor([[0.0004889113489342363, -0.7847176363008063, 0.00015172012997570039, -2.355121423735089, -0.0008362295771768507, 1.570923002322515, 0.7848479620383846, 0, 0]])
# q.requires_grad_(True)
# a,b = gt_robot_model.compute_forward_kinematics(
#             q=q, link_name="panda_virtual_ee_link"
#         )
a,b = gt_robot_model.compute_forward_kinematics(
            q=q, link_name="panda_gripper_center" # fake_target
        )
print(a)
print(b)
# a.backward(torch.ones_like(a))
# print(q.grad)

# compute Jacobian
a, b = gt_robot_model.compute_endeffector_jacobian(q, link_name="panda_gripper_center")

J_v = a[:, :, :7]
J_omega = b[:, :, :7]

print(J_v)