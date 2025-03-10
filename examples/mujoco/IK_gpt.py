import mujoco
import numpy as np
import matplotlib.pyplot as plt
import time
import glfw  # Import glfw directly instead of through mujoco.glfw

# Load simple 2-DOF robot arm XML
model_xml = """
<mujoco model="double_pendulum">
    <compiler angle="degree"/>
    <option gravity="0 0 -9.81"/>
    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
        <texture name="plane" type="2d" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" width="512" height="512"/>
        <material name="plane" texture="plane" texrepeat="1 1" texuniform="true" reflectance="0.2"/>
    </asset>
    <worldbody>
        <light diffuse="0.8 0.8 0.8" pos="0 0 3" dir="0 0 -1"/>
        <geom name="floor" type="plane" size="5 5 0.1" material="plane"/>
        
        <body name="base" pos="0 0 1">
            <geom name="base_geom" type="sphere" size="0.1" rgba="0.7 0.7 0.7 1"/>
            <body name="link1" pos="0 0 -0.5">
                <joint name="joint1" type="hinge" axis="0 0 1" pos="0 0 0" limited="true" range="-180 180"/>
                <geom name="link1_geom" type="capsule" fromto="0 0 0 0 -0.5 0" size="0.05" rgba="0.8 0.3 0.3 1"/>
                <body name="link2" pos="0 -0.5 0">
                    <joint name="joint2" type="hinge" axis="0 0 1" pos="0 0 0" limited="true" range="-180 180"/>
                    <geom name="link2_geom" type="capsule" fromto="0 0 0 0 -0.5 0" size="0.05" rgba="0.3 0.8 0.3 1"/>
                    <site name="endpoint" pos="0 -0.5 0" size="0.05" rgba="0 0 1 0.5"/>
                </body>
            </body>
        </body>
    </worldbody>

    <actuator>
        <motor joint="joint1" name="motor1" gear="100" ctrllimited="true" ctrlrange="-1 1"/>
        <motor joint="joint2" name="motor2" gear="100" ctrllimited="true" ctrlrange="-1 1"/>
    </actuator>
</mujoco>
"""

# Create the model and data objects
model = mujoco.MjModel.from_xml_string(model_xml)
data = mujoco.MjData(model)

# Convert degrees to radians for calculations (since model uses degrees)
def deg2rad(deg):
    return deg * np.pi / 180.0

def rad2deg(rad):
    return rad * 180.0 / np.pi

# Target end-effector position (in x-y plane)
TARGET_POS = np.array([0.7, 0.3, 0.1])  # Adding z-coordinate to match mujoco's 3D space

# IK Functions (working in radians)
def forward_kinematics(q):
    """ Compute end-effector position given joint angles in radians."""
    l1, l2 = 0.5, 0.5
    x = l1 * np.cos(q[0]) + l2 * np.cos(q[0] + q[1])
    y = l1 * np.sin(q[0]) + l2 * np.sin(q[0] + q[1])
    return np.array([x, y, 0.1])  # Fixed z-coordinate

def jacobian(q):
    """ Compute Jacobian matrix for 2-DOF arm."""
    l1, l2 = 0.5, 0.5
    J = np.zeros((3, 2))  # 3D position, 2 joints
    
    # x-rows
    J[0, 0] = -l1 * np.sin(q[0]) - l2 * np.sin(q[0] + q[1])
    J[0, 1] = -l2 * np.sin(q[0] + q[1])
    
    # y-rows
    J[1, 0] = l1 * np.cos(q[0]) + l2 * np.cos(q[0] + q[1])
    J[1, 1] = l2 * np.cos(q[0] + q[1])
    
    # z-rows are zero since movement is in xy plane
    
    return J

def ik_dls(q, target, lambda_=0.01, max_iter=50):
    """ Damped Least Squares IK """
    q = q.copy()  # Make a copy to avoid modifying the original
    
    for i in range(max_iter):
        current_pos = forward_kinematics(q)
        error = target - current_pos
        
        if np.linalg.norm(error) < 1e-3:
            break
            
        J = jacobian(q)
        J_dls = J.T @ np.linalg.inv(J @ J.T + lambda_ * np.eye(3))
        delta_q = J_dls @ error
        
        # Apply joint limit damping
        q += delta_q
        
    return q

def ik_jacobian_transpose(q, target, alpha=0.01, max_iter=50):
    """ Jacobian Transpose IK """
    q = q.copy()  # Make a copy to avoid modifying the original
    
    for i in range(max_iter):
        current_pos = forward_kinematics(q)
        error = target - current_pos
        
        if np.linalg.norm(error) < 1e-3:
            break
            
        J = jacobian(q)
        delta_q = alpha * J.T @ error
        
        q += delta_q
        
    return q

# Set up plotting
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
line_dls, = ax1.plot([], [], 'r-', label='DLS Error')
line_jt, = ax1.plot([], [], 'b-', label='JT Error')
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Error (m)')
ax1.set_title('IK Error Comparison')
ax1.legend()
ax1.grid(True)

scatter_target = ax2.scatter([], [], c='g', marker='*', s=100, label='Target')
scatter_dls = ax2.scatter([], [], c='r', marker='o', label='DLS')
scatter_jt = ax2.scatter([], [], c='b', marker='x', label='JT')
ax2.set_xlabel('X Position (m)')
ax2.set_ylabel('Y Position (m)')
ax2.set_title('End Effector Positions')
ax2.set_xlim(-1, 1.5)
ax2.set_ylim(-1, 1.5)
ax2.grid(True)
ax2.legend()

plt.tight_layout()

# For storing error history
error_history_dls = []
error_history_jt = []
iterations = []

try:
    # Initialize GLFW
    if not glfw.init():
        raise Exception("Failed to initialize GLFW")
    
    # Create a window and make it current
    window = glfw.create_window(1200, 900, "MuJoCo IK Demo", None, None)
    if not window:
        glfw.terminate()
        raise Exception("Failed to create GLFW window")
    
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    
    # Create camera and scene
    camera = mujoco.MjvCamera()
    option = mujoco.MjvOption()
    
    scene = mujoco.MjvScene(model, maxgeom=1000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    
    # Initialize the camera configuration
    camera.azimuth = 90
    camera.elevation = -20
    camera.distance = 2.5
    camera.lookat = np.array([0.0, 0.0, 0.0])
    
    # Set initial joint positions (in degrees for mujoco)
    q_init_deg = np.array([0.0, 0.0])
    data.qpos[0:2] = deg2rad(q_init_deg)
    mujoco.mj_forward(model, data)
    
    # Simulation loop
    for i in range(1000):
        if glfw.window_should_close(window):
            break
        
        # Get window size
        viewport_width, viewport_height = glfw.get_framebuffer_size(window)
        
        # Get current joint angles (convert to radians for IK)
        q_current_rad = np.array([data.qpos[0], data.qpos[1]])
        
        # Run IK algorithms
        q_dls_rad = ik_dls(q_current_rad, TARGET_POS)
        q_jt_rad = ik_jacobian_transpose(q_current_rad, TARGET_POS)
        
        # Convert calculated angles back to degrees for visualization
        q_dls_deg = rad2deg(q_dls_rad)
        q_jt_deg = rad2deg(q_jt_rad)
        
        # Calculate end effector positions using our FK function
        ee_dls = forward_kinematics(q_dls_rad)
        ee_jt = forward_kinematics(q_jt_rad)
        
        # Set control signal to move toward IK solution (using DLS as the controller)
        # Scale down the control to make it smoother
        control_gain = 0.05
        joint_error = q_dls_rad - q_current_rad
        data.ctrl[0:2] = control_gain * joint_error
        
        # Step the simulation
        mujoco.mj_step(model, data)
        
        # Calculate errors
        error_dls = np.linalg.norm(TARGET_POS - ee_dls)
        error_jt = np.linalg.norm(TARGET_POS - ee_jt)
        
        # Store errors for plotting
        error_history_dls.append(error_dls)
        error_history_jt.append(error_jt)
        iterations.append(i)
        
        # Update plots every 10 steps
        if i % 10 == 0:
            # Update error plot
            line_dls.set_data(iterations, error_history_dls)
            line_jt.set_data(iterations, error_history_jt)
            ax1.relim()
            ax1.autoscale_view()
            
            # Update position plot
            scatter_target.set_offsets([TARGET_POS[0], TARGET_POS[1]])
            scatter_dls.set_offsets([ee_dls[0], ee_dls[1]])
            scatter_jt.set_offsets([ee_jt[0], ee_jt[1]])
            
            plt.draw()
            plt.pause(0.001)
        
        # Update the scene and render
        mujoco.mjv_updateScene(model, data, option, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
        
        # Create a proper MjrRect for the viewport
        viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)
        
        # Render using the proper viewport object
        mujoco.mjr_render(viewport, scene, context)
        
        # Swap OpenGL buffers
        glfw.swap_buffers(window)
        
        # Process pending GUI events
        glfw.poll_events()
        
        time.sleep(0.01)  # Control simulation speed
    
except Exception as e:
    print(f"Simulation error: {e}")

finally:
    # Clean up
    plt.ioff()
    plt.show()  # Show final plots
    
    # Clean up GLFW
    try:
        glfw.terminate()
    except:
        pass