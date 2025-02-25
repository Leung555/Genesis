import genesis as gs
import numpy as np

import cv2

cv2.setNumThreads(0)
cv2.waitKey(1)

gs.init(
    backend=gs.cuda,
    precision="32",
)

scene = gs.Scene(
    show_viewer=True,
)

plane = scene.add_entity(gs.morphs.Plane())

g1 = scene.add_entity(
    gs.morphs.MJCF(
        file="urdf/g1/g1_23dof_rev_1_0.xml",
        pos   = (0.0, 0.0, 1.0),
        ),
)

scene.build()

r_arm_jnt_names = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_ankle_roll_joint",
    "right_wrist_roll_joint",
]

r_arm_dofs_idx = [g1.get_joint(name).dof_idx_local for name in r_arm_jnt_names]

g1.set_dofs_kp(
    kp=[4500] * 6,
    dofs_idx_local=r_arm_dofs_idx,
)
g1.set_dofs_kv(
    kv=[350] * 6,
    dofs_idx_local=r_arm_dofs_idx,
)

end_effector = g1.get_link("right_wrist_roll_rubber_hand")


for i in range(500):
    commands = np.random.uniform(-10, 10, size=len(r_arm_dofs_idx))
    g1.control_dofs_position(commands, dofs_idx_local=r_arm_dofs_idx)

    scene.step()
