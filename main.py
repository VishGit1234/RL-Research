import mujoco
import os
import mujoco.viewer
import random

model = mujoco.MjModel.from_xml_path(os.path.join('kinova_gen3', 'scene.xml'))
data = mujoco.MjData(model)
# renderer = mujoco.Renderer(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
  T = 10000
  # simulate
  for t in range(T - 1):
    print(data.ctrl)
    
    # step

    # get control

    data.ctrl = [1 - 2*random.random() for i in range(7)]

    mujoco.mj_step(model, data)
    # renderer.update_scene(data)
    # pixels = renderer.render()

    viewer.sync()
