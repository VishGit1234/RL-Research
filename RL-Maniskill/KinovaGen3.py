from typing import Dict, Union

import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill import ASSET_DIR, PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, DictControllerConfig, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.controllers.base_controller import ControllerConfig
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils


@register_agent()
class KinovaGen3(BaseAgent):
    uid = "kinova_gen3"
    urdf_path = f".\\kinova_gen3\\Gen3-with-gripper.urdf"
    disable_self_collisions = True
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            left_inner_finger_pad=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            right_inner_finger_pad=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
        ),
    )
    # List of real joint names from the combined URDF (arm + gripper)
    arm_joint_names = [
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
        "joint_6",
        "joint_7",
    ]
    gripper_joint_names = [
        "left_outer_knuckle_joint",
        "right_outer_knuckle_joint",
        "left_inner_knuckle_joint",
        "right_inner_knuckle_joint",
        "left_inner_finger_joint",
        "right_inner_finger_joint",
    ]
    ee_link_name = "end_effector_link"

    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100

    gripper_stiffness = 1e3
    gripper_damping = 1e2
    gripper_force_limit = 100
    
    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [1.6530, -1.1489, 7.5884, -2.2817, 1.7474, 1.1129, 6.9761, 0.8213, 0.8200, 0.8209, 0.8208, 0.8217, 0.8210]
            ),
            pose=sapien.Pose(),
        )
    )

    @property
    def _controller_configs(
        self,
    ) -> Dict[str, Union[ControllerConfig, DictControllerConfig]]:
        # Arm joint position controller
        arm_pd_joint_pos = PDJointPosControllerConfig(
            joint_names=self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        # Arm end effector delta position controller
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        # define a passive controller config to simply "turn off" other joints from being controlled and set their properties (damping/friction) to 0.
        # these joints are controlled passively by the mimic controller later on.
        passive_finger_joint_names = [
            "left_inner_knuckle_joint",
            "right_inner_knuckle_joint",
            "left_inner_finger_joint",
            "right_inner_finger_joint",
        ]
        passive_finger_joints = PassiveControllerConfig(
            joint_names=passive_finger_joint_names,
            damping=0,
            friction=0,
        )

        finger_joint_names = ["left_outer_knuckle_joint", "right_outer_knuckle_joint"]
        # use a mimic controller config to define one action to control both fingers
        finger_mimic_pd_joint_pos = PDJointPosMimicControllerConfig(
            joint_names=finger_joint_names,
            lower=None,
            upper=None,
            stiffness=1e5,
            damping=1e3,
            force_limit=0.1,
            friction=0.05,
            normalize_action=False,
        )
        finger_mimic_pd_joint_delta_pos = PDJointPosMimicControllerConfig(
            joint_names=finger_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=1e5,
            damping=1e3,
            force_limit=0.1,
            normalize_action=True,
            friction=0.05,
            use_delta=True,
        )
        return dict(
            pd_joint_pos=dict(
                arm=arm_pd_joint_pos,
                finger=finger_mimic_pd_joint_pos,
                passive_finger_joints=passive_finger_joints,
            ),
            pd_ee_delta_pose=dict(
                arm=arm_pd_ee_delta_pose,
                finger=finger_mimic_pd_joint_delta_pos,
                passive_finger_joints=passive_finger_joints,
            ),
        )

    def _after_loading_articulation(self):
        outer_finger = self.robot.active_joints_map["right_inner_finger_joint"]
        inner_knuckle = self.robot.active_joints_map["right_inner_knuckle_joint"]
        pad = outer_finger.get_child_link()
        lif = inner_knuckle.get_child_link()

        # the next 4 magic arrays come from https://github.com/haosulab/cvpr-tutorial-2022/blob/master/debug/robotiq.py which was
        # used to precompute these poses for drive creation
        p_f_right = [-1.6048949e-08, 3.7600022e-02, 4.3000020e-02]
        p_p_right = [1.3578170e-09, -1.7901104e-02, 6.5159947e-03]
        p_f_left = [-1.8080145e-08, 3.7600014e-02, 4.2999994e-02]
        p_p_left = [-1.4041154e-08, -1.7901093e-02, 6.5159872e-03]

        right_drive = self.scene.create_drive(
            lif, sapien.Pose(p_f_right), pad, sapien.Pose(p_p_right)
        )
        right_drive.set_limit_x(0, 0)
        right_drive.set_limit_y(0, 0)
        right_drive.set_limit_z(0, 0)

        outer_finger = self.robot.active_joints_map["left_inner_finger_joint"]
        inner_knuckle = self.robot.active_joints_map["left_inner_knuckle_joint"]
        pad = outer_finger.get_child_link()
        lif = inner_knuckle.get_child_link()

        left_drive = self.scene.create_drive(
            lif, sapien.Pose(p_f_left), pad, sapien.Pose(p_p_left)
        )
        left_drive.set_limit_x(0, 0)
        left_drive.set_limit_y(0, 0)
        left_drive.set_limit_z(0, 0)

    def _after_init(self):
        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )

# import mani_skill.examples.demo_robot as demo_robot_script

# demo_robot_script.main()