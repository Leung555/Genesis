import genesis as gs
import numpy as np

class SceneManager:
    def __init__(self, file_path="xml/franka_emika_panda/panda.xml"):
        """Initializes the Genesis simulation scene using an MJCF model."""
        # Initialize the Genesis environment with CUDA for acceleration
        gs.init(backend=gs.cpu, logging_level='error')
        
        # Create the scene with the viewer enabled
        self.scene = gs.Scene(show_viewer=True)

        # Load the robot from an MJCF file (XML format)
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(file=file_path)
        )

    def reset(self):
        """Reset the scene and robot to the initial state."""
        # Reset the robot joints to zero positions (can be changed based on robot's start pose)
        self.robot.control_dofs_position(np.zeros(7))  # 7-DOF position for the Panda robot (adjust as needed)
        return np.zeros(7)  # Return initial joint positions as observation

    def step(self, action):
        """Perform an action (set joint positions) and return the new robot state (observation)."""
        # Apply the action to the robot (set joint positions)
        self.robot.control_dofs_position(action)
        
        # Get and return the current joint positions as observation
        return self.robot.get_joint_positions()

    def render(self):
        """Optionally render the scene to visualize the robot's actions."""
        self.scene.step()

    def build(self):
        """Build the scene and load the robot model."""
        self.scene.build()

    def init_joint_config(self):
        self.jnt_names = [
            'joint1',
            'joint2',
            'joint3',
            'joint4',
            'joint5',
            'joint6',
            'joint7',
            'finger_joint1',
            'finger_joint2',
            ]
        self.dofs_idx = [self.robot.get_joint(name).dof_idx_local for name in self.jnt_names]

        ############ Optional: set control gains ############
        # set positional gains
        self.robot.set_dofs_kp(
            kp             = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
            dofs_idx_local = self.dofs_idx,
        )
        # set velocity gains
        self.robot.set_dofs_kv(
            kv             = np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
            dofs_idx_local = self.dofs_idx,
        )
        # set force range for safety
        self.robot.set_dofs_force_range(
            lower          = np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            upper          = np.array([ 87,  87,  87,  87,  12,  12,  12,  100,  100]),
            dofs_idx_local = self.dofs_idx,
        )