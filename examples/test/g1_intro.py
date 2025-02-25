import genesis as gs

gs.init(backend=gs.cuda)

scene = gs.Scene(show_viewer=True)
plane = scene.add_entity(gs.morphs.Plane())

g1 = scene.add_entity(
    gs.morphs.URDF(
        file='urdf/g1/g1_23dof_rev_1_0.urdf',
        pos   = (0.0, 0.0, 1.0),
        fixed = True
        ),
)

scene.build()

for i in range(500):
    scene.step()


# joint list
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

# Link list
# pelvis
# left_hip_pitch_link
# right_hip_pitch_link
# torso_link
# left_hip_roll_link
# right_hip_roll_link
# left_shoulder_pitch_link
# right_shoulder_pitch_link
# left_hip_yaw_link
# right_hip_yaw_link
# left_shoulder_roll_link
# right_shoulder_roll_link
# left_knee_link
# right_knee_link
# left_shoulder_yaw_link
# right_shoulder_yaw_link
# left_ankle_pitch_link
# right_ankle_pitch_link
# left_elbow_link
# right_elbow_link
# left_ankle_roll_link
# right_ankle_roll_link
# left_wrist_roll_rubber_hand
# right_wrist_roll_rubber_hand