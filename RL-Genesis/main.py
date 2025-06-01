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
        dt                = 0.01
    ),
)

########################## entities ##########################
plane = scene.add_entity(
    gs.morphs.Plane(),
)

# box = scene.add_entity(
#     gs.morphs.Box(
#         pos=(0.2, 0.2, 0.02),
#         size=(0.08, 0.08, 0.02)
#     ),
#     gs.materials.Rigid(
#         rho=400, # 400 kg/m^3 -> density of some types of wood
#         friction=None
#     ), # The params here can be used for domain randomization
#     # gs.surfaces.Default(
#     #     color=(196, 30, 58) # make block red
#     # )
# )

kinova_gen3 = scene.add_entity(
    gs.morphs.MJCF(file='./kinova_gen3/gen3.xml'),
)

# cam = scene.add_camera(
#     res    = (640, 480),
#     pos    = (1.5, 0.0, 2.5),
#     lookat = (0, 0, 1.0),
#     fov    = 30,
#     GUI    = False,
# )

########################## build ##########################
# create B parallel environments
B = 2
scene.build(n_envs=B, env_spacing=(3.0, 3.0))

# get the end-effector link
end_effector = kinova_gen3.get_link('bracelet_link')

# set to initial configuration
start_config = np.array([
    6.25032076, 0.60241784, 3.15709118, -2.128586102, 6.28220792, -0.39995964788322566, 1.55241801, 0.822, 0.822, 0.822, 0.822, 0.822, 0.822
])
qpos = np.tile(start_config,(B,1))

for i in range(100):
    kinova_gen3.set_dofs_position(qpos)
    scene.step()
print(end_effector.get_pos())

# print(end_effector.get_quat())

# qpos = kinova_gen3.inverse_kinematics(
#     link = end_effector,
#     pos  = np.tile(np.array([0.4, 0.0, 0.3]), (B,1)),
#     quat = np.tile(np.array([0., -1., 0., 0.]), (B,1)),
# )

# # cam.start_recording()

# qpos[:,-6:] = 0.822

# # Go to goal
# kinova_gen3.control_dofs_position(qpos)
# scene.step()


# cam.render()

# cam.stop_recording(save_to_filename='video.mp4', fps=60)


# img_arr, _, _, _ = cam.render()
# image = Image.fromarray(img_arr.astype('uint8')).convert('RGB')
# image.save('images/image.jpg')