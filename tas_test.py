import numpy as np
import torch
import random
import os

from differentiable_robot_model.robot_model import (
    DifferentiableFrankaPanda
)

torch.set_printoptions(precision=3, sci_mode=False)
random.seed(0)
np.random.seed(1)
torch.manual_seed(0)

device="cpu"
gt_robot_model = DifferentiableFrankaPanda(device=device)
#print(learnable_robot_model.print_learnable_params())
print(gt_robot_model._n_dofs)
print(gt_robot_model._name_to_idx_map)

q = torch.FloatTensor([[0, 0, 0, 0, 0, 0, 0]])
q.requires_grad_(True)
a,b = gt_robot_model.compute_forward_kinematics(
            q=q, link_name="panda_virtual_ee_link"
        )
a.backward(torch.ones_like(a))
print(q.grad)