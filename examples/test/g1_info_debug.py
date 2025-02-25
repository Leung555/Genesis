import genesis as gs

gs.init(backend=gs.cuda)

scene = gs.Scene(show_viewer=False)
plane = scene.add_entity(gs.morphs.Plane())

g1 = scene.add_entity(
    gs.morphs.URDF(file='urdf/g1/g1_23dof_rev_1_0.urdf'), #urdf/g1/g1_12dof.urdf
)

cam_0 = scene.add_camera()
scene.build()

# enter IPython's interactive mode
import IPython; IPython.embed()