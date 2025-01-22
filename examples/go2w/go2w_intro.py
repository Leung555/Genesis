import genesis as gs

gs.init(backend=gs.cuda)

scene = gs.Scene(show_viewer=True)
plane = scene.add_entity(gs.morphs.Plane())

go2w = scene.add_entity(
    gs.morphs.URDF(
        file='/home/binggwong/git/Genesis/genesis/assets/urdf/go2w_description/urdf/go2w.urdf',
        pos   = (0.0, 0.0, 0.5),
        ),
)

scene.build()

for i in range(100):
    scene.step()