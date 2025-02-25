import numpy as np
import genesis as gs
import pygame

# Function to map mouse position to target position
def map_mouse_to_commands(mouse_pos):
    x, y = mouse_pos
    # Normalize mouse position to range [-1, 1]
    x = ((x / 320.0) - 1.0) * 0.3
    y = ((y / 240.0) - 1.0) * 0.3
    return np.array([0, x, y])

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
r_arm_dofs_idx = [g1.get_joint(name).dof_idx_local for name in jnt_names]
print(r_arm_dofs_idx)

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
    kp             = np.repeat(kp_, len(r_arm_dofs_idx)),
    dofs_idx_local = r_arm_dofs_idx,
)
# set velocity gains
g1.set_dofs_kv(
    kv             = np.repeat(kv_, len(r_arm_dofs_idx)),
    dofs_idx_local = r_arm_dofs_idx,
)

########################## IK Control ##########################
# get the end-effector link
end_effector = g1.get_link('right_wrist_roll_rubber_hand')

target_pos = np.array([0, 0, 0])
target_quat = np.array([1, 0, 0, 0])
center = np.array([0.2, -0.2, 1.3])
mouse_command = np.array([0, 0, 0])

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((320, 240))

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEMOTION:
            mouse_pos = pygame.mouse.get_pos()
            mouse_command = map_mouse_to_commands(mouse_pos)
    
    target_pos = center + mouse_command
    print('target_pos:', target_pos)

    points = scene.draw_debug_points(pos=(target_pos[0], target_pos[1], target_pos[2]), radius=0.1, color=(0, 1, 0, 0.0))
    q = g1.inverse_kinematics(
        link    = end_effector,
        pos     = target_pos,
        quat    = target_quat,
        rot_mask = [False, False, True], # only restrict direction of z-axis
    )
    path = g1.plan_path(
        qpos_goal     = q,
        num_waypoints = 20, # 2s duration
    )
    # execute the planned path
    # for waypoint in path:
    #     g1.control_dofs_position(waypoint)
    #     scene.step()

    # Note that this IK is for visualization purposes, so here we do not call scene.step(), but only update the state and the visualizer
    # In actual control applications, you should instead use g1.control_dofs_position() and scene.step()
    g1.set_dofs_position(q)
    scene.visualizer.update()

pygame.quit()