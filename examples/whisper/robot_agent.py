import numpy as np

class RobotAgent:
    def __init__(self, scene_manager):
        """Initializes the robot agent with a reference to the scene manager."""
        self.scene_manager = scene_manager

    def reset(self):
        """Reset the robot state by resetting the scene."""
        return self.scene_manager.reset()

    def parse_command_to_action(self, command):
        """Converts speech command to robot action (joint positions)."""
        if "move left" in command:
            return np.array([-0.5, 0, 0, 0, 0, 0, 0])  # Example movement to the left
        elif "move right" in command:
            return np.array([0.5, 0, 0, 0, 0, 0, 0])  # Example movement to the right
        elif "pick up" in command:
            return np.array([0, 0, 0, 0, 0, 0, 0.02])  # Close gripper
        elif "release" in command:
            return np.array([0, 0, 0, 0, 0, 0, 0.08])  # Open gripper
        else:
            print(f"Unknown command: {command}")
            return np.zeros(7)  # Default action (no movement)

