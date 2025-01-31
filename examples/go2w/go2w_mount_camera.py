import numpy as np
import genesis as gs

# Define the translation matrix for z-axis translation by 0.1 units
def translate_z(units):
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, units],
        [0, 0, 0, 1]
    ])

# Define the rotation matrix for z-axis rotation by -90 degrees
def rotate_z(degrees):
    radians = np.radians(degrees)
    c, s = np.cos(radians), np.sin(radians)
    return np.array([
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

# Define the rotation matrix for y-axis rotation by 90 degrees
def rotate_y(degrees):
    radians = np.radians(degrees)
    c, s = np.cos(radians), np.sin(radians)
    return np.array([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1]
    ])

# Define the rotation matrix for x-axis rotation by 90 degrees
def rotate_x(degrees):
    radians = np.radians(degrees)
    c, s = np.cos(radians), np.sin(radians)
    return np.array([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1]
    ])

def command_action(robot, pos_arr, vel_arr):
    robot.control_dofs_position(
        np.array(list(pos_arr.values()))[:-4],
        dofs_idx[:-4],
    )
    robot.control_dofs_velocity(
        np.array(vel_arr),
        dofs_idx[-4:],
    )

########################## init ##########################
gs.init(backend=gs.gpu,
        logging_level='warning')

########################## create a scene ##########################
scene = gs.Scene(
    # viewer_options = gs.options.ViewerOptions(
    #     camera_pos    = (0, -3.5, 2.5),
    #     camera_lookat = (0.0, 0.0, 0.5),
    #     camera_fov    = 30,
    #     max_FPS       = 60,
    # ),
    sim_options=gs.options.SimOptions(
        dt=0.01,
        gravity=(0, 0, -10.0),
    ),
    show_viewer = True,
    vis_options = gs.options.VisOptions(
        show_world_frame = True, # visualize the coordinate frame of `world` at its origin
        world_frame_size = 1.0, # length of the world frame in meter
        show_link_frame  = False, # do not visualize coordinate frames of entity links
        show_cameras     = False, # do not visualize mesh and frustum of the cameras added
        plane_reflection = True, # turn on plane reflection
        ambient_light    = (0.1, 0.1, 0.1), # ambient light setting
    ),
    renderer = gs.renderers.Rasterizer(), # using rasterizer for camera rendering
)

########################## entities ##########################
plane = scene.add_entity(
    gs.morphs.Plane(),
)

sphere = scene.add_entity(
    gs.morphs.Sphere(
        pos=(1.5,0,0.6),
        radius=0.3,
    )
)

# when loading an entity, you can specify its pose in the morph.
go2w = scene.add_entity(
    gs.morphs.URDF(
        file  = '/home/binggwong/git/Genesis/genesis/assets/urdf/go2w_description/urdf/go2w.urdf',
        pos   = (0.0, 0.0, 1.0),
        euler = (0, 0, 0),
    ),
)

########################## Define Joint Name ##########################
jnt_names = [
    "FL_hip_joint",    "FR_hip_joint",    "RL_hip_joint",    "RR_hip_joint",
    "FL_thigh_joint",  "FR_thigh_joint",  "RL_thigh_joint",  "RR_thigh_joint",
    "FL_calf_joint",   "FR_calf_joint",   "RL_calf_joint",   "RR_calf_joint",
    "FR_foot_joint",   "FL_foot_joint",   "RR_foot_joint",   "RL_foot_joint",
]
dofs_idx = [go2w.get_joint(name).dof_idx_local for name in jnt_names]

default_joint_angles = {  # [rad]
    "FL_hip_joint":   0.0,    "FR_hip_joint": 0.0,    "RL_hip_joint": 0.0,    "RR_hip_joint": 0.0,
    "FL_thigh_joint": 0.8,  "FR_thigh_joint": 0.8,  "RL_thigh_joint": 1.0,  "RR_thigh_joint": 1.0,
    "FL_calf_joint": -1.5,  "FR_calf_joint": -1.5,  "RL_calf_joint": -1.5,  "RR_calf_joint": -1.5,
    "FR_foot_joint":  0.0,   "FL_foot_joint": 0.0,   "RR_foot_joint": 0.0,   "RL_foot_joint": 0.0,
}
links_names = ['base',
                'FL_hip',   'FR_hip',   'RL_hip',   'RR_hip',
                'FL_thigh', 'FR_thigh', 'RL_thigh', 'RR_thigh',
                'FL_calf',  'FR_calf',  'RL_calf',  'RR_calf',
                'FL_foot',  'FR_foot',  'RL_foot',  'RR_foot']
links_obj = [go2w.get_link(link) for link in links_names]
base = links_obj[0]

dofs_len = len(dofs_idx)
random_array = [np.random.randint(-100, 100) for _ in range(dofs_len)]
print('dofs_idx: ', dofs_idx)
print('random_array: ', random_array)
print('dofs_len: ', dofs_len)

# Control parameters
speed = 4
wheel_speed_arr = [speed] * 4

###################### camera ######################
cam = scene.add_camera(
    res    = (50, 50),
    pos    = (3.5, 0.0, 2.5),
    fov    = 60,
    GUI    = True
)

# Combine the transformations
T_translate = translate_z(0.1)
T_rotate_z = rotate_z(-90)
T_rotate_y = rotate_y(90)
T_rotate_x = rotate_x(90)

# The final transformation matrix
T_final = T_translate @ T_rotate_z @ T_rotate_x # @ T_rotate_x



# enter IPython's interactive mode
# import IPython; IPython.embed()

########################## build ##########################
scene.build()

cam.attach(base, T_final)

# render rgb, depth, segmentation mask and normal map
# rgb, depth, segmentation, normal = cam.render(depth=False, segmentation=False, normal=False)
# print(rgb)

############ Optional: set control gains ############
# set positional gains
# kp_ = 300
# kv_ = 30
# go2w.set_dofs_kp(
#     kp             = np.array([kp_, kp_, kp_, kp_,  
#                                 kp_, kp_, kp_, kp_, 
#                                 kp_, kp_, kp_, kp_,
#                                  kp_, kp_, kp_, kp_,]),
#     dofs_idx_local = dofs_idx,
# )
# # set velocity gains
# go2w.set_dofs_kv(
#     kv             = np.array([kv_, kv_, kv_, kv_,  
#                                 kv_, kv_, kv_, kv_, 
#                                 kv_, kv_, kv_, kv_,
#                                  kv_, kv_, kv_, kv_,]),
#     dofs_idx_local = dofs_idx,
# )

for i in range(-100, 600):
    if i == 0:
        sp = 10
        wheel_speed_arr = [sp] * 4
        command_action(go2w, default_joint_angles, wheel_speed_arr)
    elif i == 150:
        sp = -10
        wheel_speed_arr = [sp] * 4
        command_action(go2w, default_joint_angles, wheel_speed_arr)

    elif i == 300:
        sp = 3
        wheel_speed_arr = [sp, -sp, sp , -sp]
        command_action(go2w, default_joint_angles, wheel_speed_arr)

    elif i == 450:
        sp = 3
        wheel_speed_arr = [-sp, sp, -sp, sp]
        command_action(go2w, default_joint_angles, wheel_speed_arr)

    scene.step()

    cam.render()

# set force range for safety
# go2w.set_dofs_force_range(
#     lower          = np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
#     upper          = np.array([ 87,  87,  87,  87,  12,  12,  12,  100,  100]),
#     dofs_idx_local = dofs_idx,
# )

# # Static posture
# for i in range(250):
#     go2w.control_dofs_position(
#         np.array(list(default_joint_angles.values()))[:-4],
#         dofs_idx[:-4],
#     )
#     go2w.control_dofs_velocity(
#         np.array(wheel_speed_arr),
#         dofs_idx[-4:],
#     )

# Hard reset
# for i in range(500):
#     if i < 50:
#         go2w.set_dofs_position(random_array, dofs_idx)
#     elif i < 100:
#         go2w.set_dofs_position(random_array, dofs_idx)
#     else:
#         go2w.set_dofs_position(random_array, dofs_idx)

#     scene.step()

# PD control
# for i in range(1250):
#     if i == 0:
#         go2w.control_dofs_position(
#             np.array(random_array),
#             dofs_idx,
#         )
#     elif i == 250:
#         go2w.control_dofs_position(
#             np.array(random_array),
#             dofs_idx,
#         )
#     elif i == 500:
#         go2w.control_dofs_position(
#             np.array(random_array),
#             dofs_idx,
#         )
#     elif i == 750:
#         # control first dof with velocity, and the rest with position
#         go2w.control_dofs_position(
#             np.array(random_array)[1:],
#             dofs_idx[1:],
#         )
#         go2w.control_dofs_velocity(
#             np.array(random_array)[:1],
#             dofs_idx[:1],
#         )
#     elif i == 1000:
#         go2w.control_dofs_force(
#             np.array(random_array),
#             dofs_idx,
#         )
#     # This is the control force computed based on the given control command
#     # If using force control, it's the same as the given control command
#     print('control force:', go2w.get_dofs_control_force(dofs_idx))

#     # This is the actual force experienced by the dof
#     print('internal force:', go2w.get_dofs_force(dofs_idx))

#     scene.step()