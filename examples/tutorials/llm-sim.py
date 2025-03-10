import genesis as gs
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Initialize Genesis
gs.init(backend=gs.cpu)
scene = gs.Scene(show_viewer=True)
plane = scene.add_entity(gs.morphs.Plane())  # Ground plane

# FastAPI App
app = FastAPI()

# Define input schema
class ObjectRequest(BaseModel):
    shape: str  # e.g., "box", "sphere", "cylinder"
    position: tuple[float, float, float]
    size: tuple[float, float, float] = (1.0, 1.0, 1.0)  # Default size
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)  # Default white

# Add object function
def add_object(shape: str, position: tuple, size: tuple, color: tuple):
    if shape == "box":
        obj = gs.morphs.Box(size=size, pos=position, color=color)
    elif shape == "sphere":
        obj = gs.morphs.Sphere(radius=size[0] / 2, pos=position, color=color)
    elif shape == "cylinder":
        obj = gs.morphs.Cylinder(radius=size[0] / 2, height=size[2], pos=position, color=color)
    else:
        raise ValueError("Unsupported shape")
    
    scene.add_entity(obj)
    return {"message": f"Added {shape} at {position}"}

# API endpoint to add objects
@app.post("/add_object")
def add_object_api(request: ObjectRequest):
    try:
        return add_object(request.shape, request.position, request.size, request.color)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run Genesis simulation
scene.build()

# running = True
# while running:
#     print("Running simulation...")

