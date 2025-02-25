import torch
import jax
import numpy as np
from evosax import CMA_ES
import genesis as gs

# Initialize Genesis with GPU backend
gs.init(backend=gs.gpu, logging_level='error')

# Create the scene in Genesis
scene = gs.Scene(
    show_viewer=False,  # Hide the viewer for parallel simulations
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3.5, -1.0, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
    ),
    rigid_options=gs.options.RigidOptions(
        dt=0.01,
    ),
)

# Add a plane (ground) and the robot (Franka Emika Panda) to the scene
plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'))

# Build the environment with 20 parallel environments
B = 20  # Number of parallel environments
scene.build(n_envs=B, env_spacing=(1.0, 1.0))

# Define a simple linear model for robot control positions
class LinearModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# Instantiate the linear model
input_dim = 2  # Number of input dimensions (can be adjusted)
output_dim = 9  # Number of DOFs for the robot (Franka has 9 DOFs)
linear_model = LinearModel(input_dim, output_dim)

# Set up the Evolution Strategy (CMA-ES) to optimize the model's weights
# num_weights = input_dim * output_dim + output_dim  # Weights + Biases in the linear model
rng = jax.random.PRNGKey(0)
strategy = CMA_ES(popsize=20, num_dims=input_dim * output_dim + output_dim, elite_ratio=0.5)
es_params = strategy.default_params
state = strategy.initialize(rng, es_params)

# Define the fitness evaluation function for the population
import numpy as np
import torch

def evaluate_population(weights, scene, franka, B):
    """Evaluate the fitness of the population (robot positions) in the simulation"""
    fitness = np.zeros(B)  # Fitness array for B robots
    
    # Loop over each individual in the population (each individual has 27 weights)
    for i in range(B):
        # Convert JAX array to NumPy array and then to PyTorch tensor
        flattened_weights = torch.tensor(np.array(weights[i]), dtype=torch.float32)  # Convert to PyTorch tensor
        
        # Unpack weights into the linear model (weights + biases)
        with torch.no_grad():
            # Extract the weight parameters for the linear model
            linear_model.linear.weight.data = flattened_weights[:input_dim * output_dim].reshape(output_dim, input_dim)
            linear_model.linear.bias.data = flattened_weights[input_dim * output_dim:]

        # Generate control positions for the robot from the linear model
        control_positions = linear_model(torch.tensor(np.array([i, i]), dtype=torch.float32))  # Example: Simple input
        franka_control = control_positions[i].to(device=gs.device)  # Move to the correct device

        # Control the robot with the current candidate (positions)
        franka.control_dofs_position(franka_control)

        # Run the simulation for a fixed number of steps (e.g., 100)
        for _ in range(100):
            scene.step()

        # Get the current DOF positions of the robot
        dofs_position = franka.get_dofs_position()  # Get the current DOF position
        
        # Fitness can be based on various criteria; here we use a simple one
        fitness[i] = -np.abs(dofs_position.sum().item())  # Sum of all DOFs for fitness

    return fitness



# Run the Evolution Strategy (ask-eval-tell loop)
num_generations = 100  # Number of generations to run
for t in range(num_generations):
    rng, rng_gen, rng_eval = jax.random.split(rng, 3)
    weights, state = strategy.ask(rng_gen, state, es_params)  # Get new population of weights for the linear model

    # Evaluate the population (robot control positions)
    fitness = evaluate_population(weights, scene, franka, B)

    # Update the strategy with the new fitness values
    state = strategy.tell(weights, fitness, state, es_params)

    # Optionally, you can print the best solution found so far
    if t % 5 == 0:
        print(f"Generation {t}, Best Fitness: {state.best_fitness}")

# After training, get the best solution found
print(f"Best Member (optimized weights): {state.best_member}")
print(f"Best Fitness: {state.best_fitness}")
