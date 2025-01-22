import genesis as gs

gs.init(backend=gs.cuda)

scene = gs.Scene(show_viewer=True)
plane = scene.add_entity(gs.morphs.Plane())

go2 = scene.add_entity(
    gs.morphs.URDF(file='urdf/go2/urdf/go2.urdf'),
)

cam_0 = scene.add_camera()
scene.build()

# enter IPython's interactive mode
import IPython; IPython.embed()