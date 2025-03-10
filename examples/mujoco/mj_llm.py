import mujoco
import numpy as np
from llama_cpp import Llama

# Load Llama model (make sure you have the .gguf model downloaded)
llm = Llama(model_path="./models/llama-3.gguf")

# Create a basic MuJoCo scene with an empty world
MODEL_XML = """
<mujoco>
    <worldbody>
        <body name="ground" pos="0 0 0">
            <geom type="plane" size="5 5 0.1" />
        </body>
    </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(MODEL_XML)
data = mujoco.MjData(model)

# Function to add objects dynamically
def add_object(obj_type, pos):
    global MODEL_XML
    obj_template = f"""
    <body name="{obj_type}" pos="{pos[0]} {pos[1]} {pos[2]}">
        <geom type="{obj_type}" size="0.2"/>
    </body>
    """
    
    MODEL_XML = MODEL_XML.replace("</worldbody>", obj_template + "</worldbody>")
    global model, data
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    data = mujoco.MjData(model)
    print(f"Added {obj_type} at {pos}")

# Function to process user prompt
def process_prompt(prompt):
    response = llm(f"Extract object type and position: {prompt}")
    words = response["choices"][0]["text"].lower().split()
    
    obj_type = ""  # Default object type
    pos = [0, 0, 0]  # Default position
    
    if "sphere" in words:
        obj_type = "sphere"
    elif "cube" in words or "box" in words:
        obj_type = "box"
    
    for i, word in enumerate(words):
        if word.replace(".", "").isdigit():
            pos = [float(words[i]), float(words[i+1]), float(words[i+2])]
            break
    
    if obj_type:
        add_object(obj_type, pos)
    else:
        print("Unknown object. Try: 'Add sphere' or 'Add cube at (x, y, z)'")

# Main loop to take user input
while True:
    user_input = input("\nEnter command (e.g., 'add sphere at 1 2 0.5' or 'exit'): ").strip()
    if user_input.lower() == "exit":
        break
    process_prompt(user_input)

# Run simulation
mujoco.mj_step(model, data)
