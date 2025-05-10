import genesis as gs
import torch
from PIL import Image
import numpy as np

########################## init ##########################
gs.init(backend=gs.gpu)

########################## create a scene ##########################
scene = gs.Scene(
    show_viewer    = False,
    rigid_options = gs.options.RigidOptions(
        dt                = 0.01,
    ),
)

########################## entities ##########################
plane = scene.add_entity(
    gs.morphs.Plane(),
)

box = scene.add_entity(
    gs.morphs.Box(
        pos=(0.2, 0.2, 0.02),
        size=(0.08, 0.08, 0.02)
    ),
    gs.materials.Rigid(
        rho=400, # 400 kg/m^3 -> density of some types of wood
        friction=None
    ), # The params here can be used for domain randomization
    gs.surfaces.Default(
        color=(196, 30, 58) # make block red
    )
)

kinova_gen3 = scene.add_entity(
    gs.morphs.MJCF(file='./kinova_gen3/gen3.xml'),
)

cam = scene.add_camera(
    res    = (320, 240),
    pos    = (1.5, 0.0, 2.5),
    lookat = (0, 0, 1.0),
    fov    = 30,
    GUI    = False
)

########################## build ##########################

# create 20 parallel environments
B = 5
scene.build(n_envs=B, env_spacing=(3.0, 3.0))

# # get the end-effector link
# end_effector = kinova_gen3.get_link('bracelet_link')

# # move to pre-grasp pose
# qpos = kinova_gen3.inverse_kinematics(
#     link = end_effector,
#     pos  = np.array([0.4, 0.0, 0.3]),
#     quat = np.array([0, 1, 0, 0]),
# )
# # gripper closed pos
# qpos[-6:] = 0.822
# path = kinova_gen3.plan_path(
#     qpos_goal     = qpos,
#     num_waypoints = 200, # 2s duration
# )
# # execute the planned path
# for waypoint in path:
#     kinova_gen3.control_dofs_position(waypoint)
#     scene.step()

# allow robot to reach the last waypoint
for i in range(100):
    scene.step()

img_arr, _, _, _ = cam.render()
image = Image.fromarray(img_arr.astype('uint8')).convert('RGB')
image.save('images/image.jpg')