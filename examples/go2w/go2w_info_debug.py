import genesis as gs

gs.init(backend=gs.cuda)

scene = gs.Scene(show_viewer=False)
plane = scene.add_entity(gs.morphs.Plane())

go2w = scene.add_entity(
    gs.morphs.URDF(file='/home/binggwong/git/Genesis/genesis/assets/urdf/go2w_description/urdf/go2w.urdf'),
)

cam_0 = scene.add_camera()
scene.build()

# enter IPython's interactive mode
import IPython; IPython.embed()