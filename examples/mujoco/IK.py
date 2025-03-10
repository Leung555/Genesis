import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import mujoco
import time
import os
from datetime import datetime

# Create a temporary XML file for the MuJoCo model
XML_TEMPLATE = """
<mujoco model="3dof_arm">
  <option timestep="0.002" integrator="Euler" gravity="0 0 0"/>
  <worldbody>
    <geom name="ground" type="plane" size="1 1 0.1" rgba="0.3 0.3 0.3 1"/>
    <light name="light" pos="0 0 3" dir="0 0 -1" diffuse="0.5 0.5 0.5"/>
    
    <body name="base" pos="0 0 0.1">
      <joint name="joint0" type="hinge" axis="0 0 1" pos="0 0 0"/>
      <geom name="link0" type="cylinder" size="0.05 0.05" rgba="0.7 0.7 0.7 1"/>
      
      <body name="link1" pos="0.1 0 0">
        <joint name="joint1" type="hinge" axis="0 0 1" pos="0 0 0"/>
        <geom name="link1_geom" type="capsule" size="0.04 0.4" pos="0.4 0 0" quat="0.707 0 0.707 0" rgba="0 0.7 1 1"/>
        
        <body name="link2" pos="0.8 0 0">
          <joint name="joint2" type="hinge" axis="0 0 1" pos="0 0 0"/>
          <geom name="link2_geom" type="capsule" size="0.03 0.3" pos="0.3 0 0" quat="0.707 0 0.707 0" rgba="0 0.5 1 1"/>
          
          <body name="link3" pos="0.6 0 0">
            <joint name="joint3" type="hinge" axis="0 0 1" pos="0 0 0"/>
            <geom name="link3_geom" type="capsule" size="0.02 0.2" pos="0.2 0 0" quat="0.707 0 0.707 0" rgba="0 0.3 1 1"/>
            
            <site name="end_effector" pos="0.4 0 0" size="0.04" rgba="1 0 0 1"/>
          </body>
        </body>
      </body>
    </body>
    
    <site name="target" pos="1.0 0.5 0.1" size="0.05" rgba="0 1 0 1"/>
  </worldbody>
  
  <actuator>
    <motor joint="joint0" name="motor0" gear="100" ctrllimited="true" ctrlrange="-1 1"/>
    <motor joint="joint1" name="motor1" gear="100" ctrllimited="true" ctrlrange="-1 1"/>
    <motor joint="joint2" name="motor2" gear="100" ctrllimited="true" ctrlrange="-1 1"/>
    <motor joint="joint3" name="motor3" gear="100" ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
  
  <sensor>
    <jointpos name="joint0_pos" joint="joint0"/>
    <jointpos name="joint1_pos" joint="joint1"/>
    <jointpos name="joint2_pos" joint="joint2"/>
    <jointpos name="joint3_pos" joint="joint3"/>
    <framepos name="end_effector_pos" objtype="site" objname="end_effector"/>
    <framepos name="target_pos" objtype="site" objname="target"/>
  </sensor>
</mujoco>
"""

class RobotArmSimulation:
    def __init__(self):
        # Create temporary XML file
        self.xml_path = "temp_robot_arm.xml"
        with open(self.xml_path, "w") as f:
            f.write(XML_TEMPLATE)
        
        # Load model and create data
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Initialize parameters
        self.num_joints = 4
        self.dof = self.num_joints
        self.target_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target")
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
        
        # Set initial joint positions
        self.data.qpos[:self.dof] = np.zeros(self.dof)
        mujoco.mj_forward(self.model, self.data)
        
        # Metrics storage
        self.jacobian_errors = []
        self.ccd_errors = []
        self.jacobian_times = []
        self.ccd_times = []
        self.timestamps = []
    
    def __del__(self):
        # Clean up temporary XML file
        if os.path.exists(self.xml_path):
            os.remove(self.xml_path)
    
    def set_target_position(self, pos):
        """Set target site position"""
        self.model.site_pos[self.target_site_id] = pos
    
    def get_end_effector_position(self):
        """Get current end effector position"""
        mujoco.mj_forward(self.model, self.data)
        return self.data.site_xpos[self.ee_site_id].copy()
    
    def get_target_position(self):
        """Get target position"""
        return self.data.site_xpos[self.target_site_id].copy()
    
    def get_joint_positions(self):
        """Get current joint positions"""
        return self.data.qpos[:self.dof].copy()
    
    def set_joint_positions(self, q):
        """Set joint positions"""
        self.data.qpos[:self.dof] = q
        mujoco.mj_forward(self.model, self.data)
    
    def calculate_jacobian(self):
        """Calculate the Jacobian matrix for the end effector"""
        # Get end effector site ID
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        
        # Get site Jacobian (position and rotation)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
        
        # We only need position Jacobian for the active joints
        return jacp[:, :self.dof]
    
    def jacobian_transpose_ik(self, target_pos, max_iter=100, alpha=0.01, tolerance=1e-4):
        """Jacobian Transpose IK method"""
        start_time = time.time()
        
        # Get initial joint positions
        q = self.get_joint_positions()
        
        for i in range(max_iter):
            # Get current end effector position
            ee_pos = self.get_end_effector_position()
            
            # Calculate error
            error = target_pos - ee_pos
            error_norm = np.linalg.norm(error)
            
            # Check if we've reached the target
            if error_norm < tolerance:
                break
            
            # Calculate Jacobian
            J = self.calculate_jacobian()
            
            # Calculate joint update using Jacobian transpose
            dq = alpha * J.T @ error
            
            # Update joint positions
            q = q + dq
            
            # Apply joint positions
            self.set_joint_positions(q)
        
        end_time = time.time()
        computation_time = (end_time - start_time) * 1000  # in milliseconds
        
        return q, error_norm, computation_time
    
    def cyclic_coordinate_descent_ik(self, target_pos, max_iter=10, iterations_per_joint=5, tolerance=1e-4):
        """Cyclic Coordinate Descent IK method"""
        start_time = time.time()
        
        # Get initial joint positions
        q = self.get_joint_positions()
        
        for iteration in range(max_iter):
            # Get current end effector position
            ee_pos = self.get_end_effector_position()
            
            # Calculate error
            error = target_pos - ee_pos
            error_norm = np.linalg.norm(error)
            
            # Check if we've reached the target
            if error_norm < tolerance:
                break
            
            # Iterate through each joint, starting from the last one
            for joint_idx in reversed(range(self.dof)):
                # Save original joint positions
                original_q = q.copy()
                
                # For this joint, try different angles to find the best one
                best_error = error_norm
                best_angle = q[joint_idx]
                
                for _ in range(iterations_per_joint):
                    # Get the axis of rotation for this joint (always Z-axis in this model)
                    axis = np.array([0, 0, 1])
                    
                    # Get the current end effector position
                    ee_pos = self.get_end_effector_position()
                    
                    # Get the joint position
                    joint_pos = np.zeros(3)
                    if joint_idx == 0:
                        joint_pos = np.array([0, 0, 0.1])  # Base joint
                    elif joint_idx == 1:
                        joint_pos = np.array([0.1, 0, 0.1])  # First joint
                    elif joint_idx == 2:
                        joint_pos = np.array([0.9, 0, 0.1])  # Second joint
                    elif joint_idx == 3:
                        joint_pos = np.array([1.5, 0, 0.1])  # Third joint
                    
                    # Calculate vectors
                    joint_to_ee = ee_pos - joint_pos
                    joint_to_target = target_pos - joint_pos
                    
                    # Calculate the angle between them
                    # Project onto the XY plane since rotation is around Z
                    joint_to_ee_xy = joint_to_ee.copy()
                    joint_to_ee_xy[2] = 0
                    joint_to_target_xy = joint_to_target.copy()
                    joint_to_target_xy[2] = 0
                    
                    # Normalize vectors
                    if np.linalg.norm(joint_to_ee_xy) > 0:
                        joint_to_ee_xy = joint_to_ee_xy / np.linalg.norm(joint_to_ee_xy)
                    if np.linalg.norm(joint_to_target_xy) > 0:
                        joint_to_target_xy = joint_to_target_xy / np.linalg.norm(joint_to_target_xy)
                    
                    # Calculate dot product and cross product
                    dot_product = np.dot(joint_to_ee_xy, joint_to_target_xy)
                    # Ensure dot product is within valid range for arccos
                    dot_product = np.clip(dot_product, -1.0, 1.0)
                    
                    # Calculate angle
                    angle_change = np.arccos(dot_product)
                    
                    # Determine sign of angle change
                    cross_product = np.cross(joint_to_ee_xy, joint_to_target_xy)
                    if cross_product[2] < 0:
                        angle_change = -angle_change
                    
                    # Apply angle change
                    q[joint_idx] += angle_change
                    self.set_joint_positions(q)
                    
                    # Evaluate new error
                    new_ee_pos = self.get_end_effector_position()
                    new_error = np.linalg.norm(target_pos - new_ee_pos)
                    
                    # Save if better
                    if new_error < best_error:
                        best_error = new_error
                        best_angle = q[joint_idx]
                    
                # Set the joint to its best angle
                q[joint_idx] = best_angle
                self.set_joint_positions(q)
        
        end_time = time.time()
        computation_time = (end_time - start_time) * 1000  # in milliseconds
        
        return q, error_norm, computation_time
    
    def run_comparison(self, num_targets=50, visualize=True, save_metrics=True):
        """Run a comparison between Jacobian Transpose and CCD methods"""
        # Setup for visualization if enabled
        use_viewer = False
        
        if visualize:
            try:
                # Try to import the newer mujoco viewer
                import mujoco.viewer
                use_viewer = True
            except ImportError:
                print("MuJoCo viewer module not available. Running without visualization.")
                use_viewer = False
        
        # Create viewer if visualizing
        viewer = None
        if use_viewer:
            try:
                viewer = mujoco.viewer.launch(self.model, self.data)
                # Set camera to a better position
                with viewer.lock():
                    viewer.opt.distance = 4
                    viewer.opt.azimuth = 140
                    viewer.opt.elevation = -20
            except Exception as e:
                print(f"Could not initialize viewer: {e}")
                use_viewer = False
        
        # Reset metrics
        self.jacobian_errors = []
        self.ccd_errors = []
        self.jacobian_times = []
        self.ccd_times = []
        self.timestamps = []
        
        # Setup target trajectory
        radius = 0.7
        center = np.array([0.6, 0, 0.1])
        target_positions = []
        
        for i in range(num_targets):
            angle = 2 * np.pi * i / num_targets
            pos = center + radius * np.array([np.cos(angle), np.sin(angle), 0])
            target_positions.append(pos)
        
        # Create a separate model and data for each method to avoid interference
        jacobian_model = mujoco.MjModel.from_xml_path(self.xml_path)
        jacobian_data = mujoco.MjData(jacobian_model)
        
        ccd_model = mujoco.MjModel.from_xml_path(self.xml_path)
        ccd_data = mujoco.MjData(ccd_model)
        
        start_time = time.time()
        
        for i, target_pos in enumerate(target_positions):
            print(f"Target {i+1}/{num_targets}")
            
            # Set target position
            jacobian_model.site_pos[self.target_site_id] = target_pos
            ccd_model.site_pos[self.target_site_id] = target_pos
            
            # Reset joint positions
            initial_q = np.array([0.0, 0.0, 0.0, 0.0])
            
            # Jacobian Transpose IK
            jacobian_data.qpos[:self.dof] = initial_q
            mujoco.mj_forward(jacobian_model, jacobian_data)
            
            # Create a mini-class to match the interface for Jacobian method
            class JacobianSimulation:
                def __init__(self, model, data, dof, ee_site_id):
                    self.model = model
                    self.data = data
                    self.dof = dof
                    self.ee_site_id = ee_site_id
                
                def get_end_effector_position(self):
                    mujoco.mj_forward(self.model, self.data)
                    return self.data.site_xpos[self.ee_site_id].copy()
                
                def get_joint_positions(self):
                    return self.data.qpos[:self.dof].copy()
                
                def set_joint_positions(self, q):
                    self.data.qpos[:self.dof] = q
                    mujoco.mj_forward(self.model, self.data)
                
                def calculate_jacobian(self):
                    jacp = np.zeros((3, self.model.nv))
                    jacr = np.zeros((3, self.model.nv))
                    mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
                    return jacp[:, :self.dof]
            
            j_sim = JacobianSimulation(jacobian_model, jacobian_data, self.dof, self.ee_site_id)
            jacobian_q, jacobian_error, jacobian_time = self.jacobian_transpose_ik(target_pos)
            
            # Apply Jacobian solution to the data
            jacobian_data.qpos[:self.dof] = jacobian_q
            mujoco.mj_forward(jacobian_model, jacobian_data)
            
            # Visualize Jacobian solution if viewer is available
            if use_viewer and viewer is not None:
                with viewer.lock():
                    # Copy data from jacobian simulation
                    self.data.qpos[:] = jacobian_data.qpos[:]
                    mujoco.mj_forward(self.model, self.data)
                # Allow time to view result
                time.sleep(0.5)
            
            # CCD IK
            ccd_data.qpos[:self.dof] = initial_q
            mujoco.mj_forward(ccd_model, ccd_data)
            
            # Create a mini-class for CCD
            class CCDSimulation:
                def __init__(self, model, data, dof, ee_site_id):
                    self.model = model
                    self.data = data
                    self.dof = dof
                    self.ee_site_id = ee_site_id
                
                def get_end_effector_position(self):
                    mujoco.mj_forward(self.model, self.data)
                    return self.data.site_xpos[self.ee_site_id].copy()
                
                def get_joint_positions(self):
                    return self.data.qpos[:self.dof].copy()
                
                def set_joint_positions(self, q):
                    self.data.qpos[:self.dof] = q
                    mujoco.mj_forward(self.model, self.data)
            
            ccd_sim = CCDSimulation(ccd_model, ccd_data, self.dof, self.ee_site_id)
            ccd_q, ccd_error, ccd_time = self.cyclic_coordinate_descent_ik(target_pos)
            
            # Apply CCD solution to the data
            ccd_data.qpos[:self.dof] = ccd_q
            mujoco.mj_forward(ccd_model, ccd_data)
            
            # Save metrics
            self.jacobian_errors.append(jacobian_error)
            self.ccd_errors.append(ccd_error)
            self.jacobian_times.append(jacobian_time)
            self.ccd_times.append(ccd_time)
            self.timestamps.append(time.time() - start_time)
            
            # Visualize CCD solution if viewer is available
            if use_viewer and viewer is not None:
                with viewer.lock():
                    # Copy data from CCD simulation
                    self.data.qpos[:] = ccd_data.qpos[:]
                    mujoco.mj_forward(self.model, self.data)
                # Allow time to view result
                time.sleep(0.5)
        
        # Close viewer
        if use_viewer and viewer is not None:
            viewer.close()
        
        # Save metrics to CSV
        if save_metrics:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ik_comparison_metrics_{timestamp}.csv"
            with open(filename, "w") as f:
                f.write("timestamp,jacobian_error,ccd_error,jacobian_time_ms,ccd_time_ms\n")
                for i in range(len(self.timestamps)):
                    f.write(f"{self.timestamps[i]},{self.jacobian_errors[i]},{self.ccd_errors[i]},{self.jacobian_times[i]},{self.ccd_times[i]}\n")
            print(f"Metrics saved to {filename}")
        
        return {
            "jacobian_errors": self.jacobian_errors,
            "ccd_errors": self.ccd_errors,
            "jacobian_times": self.jacobian_times,
            "ccd_times": self.ccd_times,
            "timestamps": self.timestamps
        }
    
    def plot_metrics(self, metrics=None):
        """Plot the comparison metrics"""
        if metrics is None:
            metrics = {
                "jacobian_errors": self.jacobian_errors,
                "ccd_errors": self.ccd_errors,
                "jacobian_times": self.jacobian_times,
                "ccd_times": self.ccd_times,
                "timestamps": self.timestamps
            }
        
        # Create figure
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot errors
        axs[0].plot(metrics["timestamps"], metrics["jacobian_errors"], 'b-', label='Jacobian Transpose')
        axs[0].plot(metrics["timestamps"], metrics["ccd_errors"], 'g-', label='CCD')
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Error (m)')
        axs[0].set_title('End Effector Position Error')
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot computation times
        axs[1].plot(metrics["timestamps"], metrics["jacobian_times"], 'b-', label='Jacobian Transpose')
        axs[1].plot(metrics["timestamps"], metrics["ccd_times"], 'g-', label='CCD')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Computation Time (ms)')
        axs[1].set_title('IK Computation Time')
        axs[1].legend()
        axs[1].grid(True)
        
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ik_comparison_plot_{timestamp}.png"
        plt.savefig(filename, dpi=300)
        print(f"Plot saved to {filename}")
        
        plt.show()

def create_animation(metrics, save_animation=True):
    """Create an animation of the metrics over time"""
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    
    # Set up line objects
    lines = []
    lines.append(axs[0].plot([], [], 'b-', label='Jacobian Transpose')[0])
    lines.append(axs[0].plot([], [], 'g-', label='CCD')[0])
    lines.append(axs[1].plot([], [], 'b-', label='Jacobian Transpose')[0])
    lines.append(axs[1].plot([], [], 'g-', label='CCD')[0])
    
    # Set up axes
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Error (m)')
    axs[0].set_title('End Effector Position Error')
    axs[0].legend()
    axs[0].grid(True)
    
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Computation Time (ms)')
    axs[1].set_title('IK Computation Time')
    axs[1].legend()
    axs[1].grid(True)
    
    # Set up animation target
    target = Circle((0, 0), 0.1, color='r', alpha=0.5)
    axs[0].add_patch(target)
    
    # Set limits
    max_time = max(metrics["timestamps"])
    axs[0].set_xlim(0, max_time)
    axs[0].set_ylim(0, max(max(metrics["jacobian_errors"]), max(metrics["ccd_errors"])) * 1.1)
    
    axs[1].set_xlim(0, max_time)
    axs[1].set_ylim(0, max(max(metrics["jacobian_times"]), max(metrics["ccd_times"])) * 1.1)
    
    def init():
        for line in lines:
            line.set_data([], [])
        target.center = (0, 0)
        return lines + [target]
    
    def animate(i):
        # Display up to frame i
        times = metrics["timestamps"][:i+1]
        
        # Update error lines
        lines[0].set_data(times, metrics["jacobian_errors"][:i+1])
        lines[1].set_data(times, metrics["ccd_errors"][:i+1])
        
        # Update computation time lines
        lines[2].set_data(times, metrics["jacobian_times"][:i+1])
        lines[3].set_data(times, metrics["ccd_times"][:i+1])
        
        # Update target
        if i < len(times):
            target.center = (times[i], 0)
        
        return lines + [target]
    
    ani = FuncAnimation(fig, animate, frames=len(metrics["timestamps"]),
                        init_func=init, blit=True, interval=50)
    
    plt.tight_layout()
    
    if save_animation:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ik_comparison_animation_{timestamp}.mp4"
        ani.save(filename, writer='ffmpeg', fps=30, dpi=200)
        print(f"Animation saved to {filename}")
    
    plt.show()
    
    return ani

def main():
    # Create the simulation
    sim = RobotArmSimulation()
    
    # Run the comparison
    print("Running comparison...")
    metrics = sim.run_comparison(num_targets=20, visualize=True)
    
    # Plot the metrics
    print("Plotting metrics...")
    # sim.plot_metrics(metrics)
    
    # Create animation
    print("Creating animation...")
    create_animation(metrics)
    
    # Clean up
    del sim
    print("Done!")

if __name__ == "__main__":
    main()