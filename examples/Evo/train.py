import genesis as gs
import torch
import jax
from evosax import CMA_ES
import numpy as np

# Initialize Genesis with GPU backend
gs.init(backend=gs.gpu,
        logging_level = 'error',)

# Create the scene in Genesis
scene = gs.Scene(
    show_viewer=False,
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

# Set up the Evolution Strategy (CMA-ES)
rng = jax.random.PRNGKey(0)
strategy = CMA_ES(popsize=20, num_dims=2, elite_ratio=0.5)  # Adjust dims based on robot control space
es_params = strategy.default_params
state = strategy.initialize(rng, es_params)

# Define the fitness evaluation function for the population
def evaluate_population(positions, scene, franka, B):
    """Evaluate the fitness of the population (robot positions) in the simulation"""
    fitness = np.zeros(B)  # Fitness array for B robots
    for i in range(B):
        # Convert JAX array to a PyTorch tensor for Genesis compatibility
        franka_control = torch.tensor([positions[i, 0], positions[i, 1], 0, -1.0, 0, 0, 0, 0.02, 0.02], device=gs.device)

        # Control the robot with the current candidate (positions)
        franka.control_dofs_position(franka_control)

        # Run the simulation for a fixed number of steps (e.g., 100)
        for _ in range(100):
            scene.step()

        # Get the current DOF positions of the robot
        dofs_position = franka.get_dofs_position()  # Get the current DOF position
        
        # Fitness can be based on various criteria; here we use a simple one
        # For example, fitness could be based on how much the robot moves forward
        # In this case, a mock fitness based on the sum of the joint positions
        fitness[i] = -np.abs(dofs_position.sum().item())  # Sum of all DOFs for fitness


    return fitness

# Run the Evolution Strategy (ask-eval-tell loop)
num_generations = 1000  # Number of generations to run
for t in range(num_generations):
    rng, rng_gen, rng_eval = jax.random.split(rng, 3)
    x, state = strategy.ask(rng_gen, state, es_params)  # Get new population of solutions (robot DOFs)

    # Convert JAX array to NumPy (needed for our fitness evaluation function)
    positions = np.array(x)

    # Evaluate the population in the simulation (fitness)
    fitness = evaluate_population(positions, scene, franka, B)

    # Update the strategy with the new fitness values
    state = strategy.tell(x, fitness, state, es_params)

    # Optionally, you can print the best solution found so far
    if t % 10 == 0:
        print(f"Generation {t}, Best Fitness: {state.best_fitness}")

# After training, get the best solution found
print(f"Best Member: {state.best_member}")
print(f"Best Fitness: {state.best_fitness}")
