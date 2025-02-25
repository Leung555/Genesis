import numpy as np
import genesis as gs

########################## init ##########################
gs.init(backend=gs.cuda,
        logging_level='error')

########################## create a scene ##########################
scene = gs.Scene(show_viewer=True)

########################## entities ##########################
plane = scene.add_entity(gs.morphs.Plane())

g1 = scene.add_entity(
    gs.morphs.URDF(
        file='urdf/g1/g1_23dof_rev_1_0.urdf',
        pos   = (0.0, 0.0, 1.0),
        fixed = True
        ),
)

jnt_names = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_ankle_roll_joint",
    "right_wrist_roll_joint"
]
R_arm_dofs_idx = [g1.get_joint(name).dof_idx_local for name in jnt_names]
print(R_arm_dofs_idx)

# two target links for visualization
target_left = scene.add_entity(
    gs.morphs.Mesh(
        file='meshes/axis.obj',
        scale=0.1,
    ),
    surface=gs.surfaces.Default(color=(1, 0.5, 0.5, 1)),
)
target_right = scene.add_entity(
    gs.morphs.Mesh(
        file='meshes/axis.obj',
        scale=0.1,
    ),
    surface=gs.surfaces.Default(color=(0.5, 1.0, 0.5, 1)),
)

########################## build ##########################
scene.build()

# Optional: set control gains ############
# set positional gains
kp_ = 500
kv_ = 30
g1.set_dofs_kp(
    kp             = np.repeat(kp_, len(R_arm_dofs_idx)),
    dofs_idx_local = R_arm_dofs_idx,
)
# set velocity gains
g1.set_dofs_kv(
    kv             = np.repeat(kv_, len(R_arm_dofs_idx)),
    dofs_idx_local = R_arm_dofs_idx,
)

########################## IK Control ##########################
# get the end-effector link
end_effector = g1.get_link('right_wrist_roll_rubber_hand')

target_quat = np.array([1, 0, 0, 0])
center = np.array([0.2, -0.2, 1.3])
r = 0.01

for i in range(0, 2000):
    target_pos_left = center + np.array([np.cos(i/360*np.pi), np.sin(i/360*np.pi), 0]) * r
    target_pos_right = target_pos_left + np.array([0.0, 0.03, 0])
    # print('target_left: ', target_pos_left)

    # target_left.set_qpos(np.concatenate([target_pos_left, target_quat]))
    # target_right.set_qpos(np.concatenate([target_pos_right, target_quat]))
    
    q = g1.inverse_kinematics(
        link    = end_effector,
        pos     = target_pos_left,
        quat    = target_quat,
        rot_mask = [False, False, True], # only restrict direction of z-axis
    )
    path = g1.plan_path(
        qpos_goal     = q,
        num_waypoints = 1, # 2s duration
    )
    # execute the planned path
    for waypoint in path:
        # print('waypoint: ', waypoint)
        # print('waypoint.shape: ', waypoint.shape)
        g1.control_dofs_position(waypoint)
        scene.step()

    # Note that this IK is for visualization purposes, so here we do not call scene.step(), but only update the state and the visualizer
    # In actual control applications, you should instead use g1.control_dofs_position() and scene.step()
    # g1.set_dofs_position(q)
    # scene.visualizer.update()

# # Heat curve
# t = np.linspace(0, np.pi, 100)  # 20 points from 0 to π
# y = 16 * np.sin(t)**3 * 0.1
# z = 13 * (np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)) * 0.1 

# # Create array of points [x, y, z]
# heart_points = np.column_stack((np.zeros_like(y), y, z))

# # Line z axis
# step_size = 100
# # x = np.linspace(-0.6, 0.6, step_size)  # 20 points along the Z-axis from -10 to 10
# # z = np.zeros_like(x)  # X stays at 0
# # y = np.zeros_like(z)  # Y stays at 0

# # line_points = np.column_stack((x, y, z))
# for i in range(step_size):
#     target_pos = center + heart_points[i] * r
#     print('target_pos: ', target_pos)

#     # move to pre-grasp pose
#     qpos = g1.inverse_kinematics(
#         link = end_effector,
#         pos  = target_pos,
#         quat = np.array([1, 0, 0, 0]),
#         rot_mask = [True, False, False], # only restrict direction of z-axis
#     )
#     # gripper open pos
#     # qpos[-2:] = 1.0
#     path = g1.plan_path(
#         qpos_goal     = qpos,
#         num_waypoints = 20, # 2s duration
#     )
#     # execute the planned path
#     for waypoint in path:
#         # print('waypoint: ', waypoint)
#         # print('waypoint.shape: ', waypoint.shape)
#         g1.control_dofs_position(waypoint[R_arm_dofs_idx], dofs_idx_local=R_arm_dofs_idx)
#         scene.step()

#     # allow g1 to reach the last waypoint
#     scene.step()


# joint_pelvis
# left_hip_pitch_joint
# right_hip_pitch_joint
# waist_yaw_joint
# left_hip_roll_joint
# right_hip_roll_joint
# left_shoulder_pitch_joint
# right_shoulder_pitch_joint
# left_hip_yaw_joint
# right_hip_yaw_joint
# left_shoulder_roll_joint
# right_shoulder_roll_joint
# left_knee_joint
# right_knee_joint
# left_shoulder_yaw_joint
# right_shoulder_yaw_joint
# left_ankle_pitch_joint
# right_ankle_pitch_joint
# left_elbow_joint
# right_elbow_joint
# left_ankle_roll_joint
# right_ankle_roll_joint
# left_wrist_roll_joint
# right_wrist_roll_joint