import rospy
from std_msgs.msg import String

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

device="cpu"
gt_robot_model = DifferentiableFrankaPanda(device=device)
#print(learnable_robot_model.print_learnable_params())
# print(gt_robot_model._n_dofs)
# print(gt_robot_model._name_to_idx_map)

# q = torch.FloatTensor([[0, 0, 0, 0, 0, 0, 0, 0, 0]])
# q = torch.FloatTensor([[-0.000221715449139051, -0.7847544138724399, 0.00010503894092899076, -2.3551128582357386, -4.516856144455796e-05, 1.5710171426932014, 0.784951122533256, 0, 0.05]])
# q = torch.FloatTensor([[-0.41615460004618293, 0.2536675692106548, -0.11877543107982268, -1.281291567250302, 1.0011549899256862, 2.3353868778729256, 0.6946935629604591, 0, 0]])
# q = torch.FloatTensor([[0.0004889113489342363, -0.7847176363008063, 0.00015172012997570039, -2.355121423735089, -0.0008362295771768507, 1.570923002322515, 0.7848479620383846, 0, 0]])
# q.requires_grad_(True)
# a,b = gt_robot_model.compute_forward_kinematics(
#             q=q, link_name="panda_virtual_ee_link"
#         )
# a,b = gt_robot_model.compute_forward_kinematics(
#             q=q, link_name="panda_gripper_center" # fake_target
#         )
# print(a)
# print(b)
# a.backward(torch.ones_like(a))
# print(q.grad)

# compute Jacobian
q_d = torch.zeros([1,9], dtype=torch.float32)

def read_join_angles(data):
    q_string = data.data[1:-1]
    # print(q_string)
    q_np = np.fromstring(q_string, sep=',')
    q = torch.zeros([1,9], dtype=torch.float32)
    q[:,:7] = torch.FloatTensor(q_np)
    # print('angles')
    # print(q)
    #print(q[:,:7])
    a, b = gt_robot_model.compute_endeffector_jacobian(q, link_name="panda_gripper_center")
    # a, b = gt_robot_model.compute_endeffector_jacobian(q, link_name="panda_gripper_center")

    # J_v = a[:, :, :7]
    # J_omega = b[:, :, :7]

    # print(J_omega)

    

    J = torch.hstack([a, b])
    # print(J)
    # print(J.shape)
    # print(q_d.squeeze(0).unsqueeze(-1).shape)
    ## cartesian_speed = torch.matmul(J.squeeze(0),q_d.squeeze(0).unsqueeze(-1))
    
    ##  print(cartesian_speed)
    A = torch.bmm(J, torch.transpose(J,2,1))
    A_inv = torch.linalg.inv(A)
    # print(A_inv)
    #det_A_inv = torch.linalg.det(A_inv)
    det_A_inv = torch.linalg.det(A)
    # print(torch.sqrt(det_A_inv))
    print(det_A_inv)
    # print(J.shape)

def calculate_jacobian():
    q = torch.zeros([1,9], dtype=torch.float32)
    a, b = gt_robot_model.compute_forward_kinematics(q=q, link_name="panda_gripper_center")
    print('FK:')
    print('Translation: ')
    print(a)
    print("Orientation: ")
    print(b)

    #a, b = gt_robot_model.compute_endeffector_jacobian(q, link_name="panda_gripper_center")
    a, b = gt_robot_model.compute_endeffector_jacobian(q, link_name="panda_link8")
    J = torch.hstack([a, b])
    print("Jacobian: ")
    print(J)


def read_joint_speed(data):
    q_string = data.data[1:-1]
    # print(q_string)
    q_np = np.fromstring(q_string, sep=',')
    q_d[:,:7] = torch.FloatTensor(q_np)
    # print('speed')
    # print(q_d)
    


def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("robot_joints", String, read_join_angles)
    rospy.Subscriber("robot_joints_speed", String, read_joint_speed)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()


