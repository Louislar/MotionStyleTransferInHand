from bvh import Bvh
from bvh_npy import Bvh_npy
import numpy as np

'''
copy from https://github.com/Mathux/ACTOR/blob/d3b0afe674e01fa2b65c89784816c3435df0a9a5/src/visualize/anim.py#L52
'''
def plot_3d_motion(motion, length, kinematic_tree_in, title="", interval=10):
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: F401
    from matplotlib.animation import FuncAnimation, writers  # noqa: F401
    #matplotlib.use('Agg')

    fig = plt.figure(figsize=[10, 5])
    ax = fig.add_subplot(111, projection='3d')
    # ax = p3.Axes3D(fig)
    # ax = fig.gca(projection='3d')

    def init():
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        ax.set_xlim(-30.0, 35.0)
        ax.set_ylim(-30.0, 35.0)
        ax.set_zlim(-30.0, 35.0)

        ax.view_init(azim=-90, elev=110)
        # ax.set_axis_off()
        ax.xaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.25)
        ax.yaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.25)
        ax.zaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.25)

    colors = ['red', 'magenta', 'black', 'green', 'blue']

    """
    Debug: to rotate the bodies
    import src.utils.rotation_conversions as geometry
    glob_rot = [0, 1.5707963267948966, 0]
    global_orient = torch.tensor(glob_rot)
    rotmat = geometry.axis_angle_to_matrix(global_orient)
    motion = np.einsum("ikj,ko->ioj", motion, rotmat)
    """
    kinematic_tree=kinematic_tree_in

    def update(index):
        ax.lines = []
        ax.collections = []
        if kinematic_tree is not None:
            for chain, color in zip(kinematic_tree, colors):
                ax.plot(motion[index, chain, 0],
                        motion[index, chain, 1],
                        motion[index, chain, 2], linewidth=4.0, color=color)
        else:
            ax.scatter(motion[1:, 0, index], motion[1:, 1, index],
                       motion[1:, 2, index], c="red")
            ax.scatter(motion[:1, 0, index], motion[:1, 1, index],
                       motion[:1, 2, index], c="blue")

    ax.set_title(title)

    ani = FuncAnimation(fig, update, frames=length, interval=interval, repeat=False, init_func=init)

    plt.tight_layout()
    # pillow have problem droping frames
    # ani.save(save_path, writer='ffmpeg', fps=1000/interval)
    plt.show()
    plt.close()

if __name__=="__main__": 
    motion=None
    with open('angry_01_000.bvh') as f:
        motion = Bvh(f.read())
    
    # Get root information
    rootName = [str(item) for item in motion.root][1].strip('ROOT ')
    print(rootName)
    print([str(item) for item in motion.root])
    
    # Get children
    children = motion.joint_direct_children(rootName)
    print(children)

    # Get all joints names
    jointsNames = motion.get_joints_names()
    print(jointsNames)
    print(len(jointsNames))

    # Get frames
    frames = motion.nframes
    print(frames)

    # Get all joints channels in first frame
    jointValues = []
    for _aJointName in jointsNames: 
        # print(_aJointName, ': ', motion.joint_channels(_aJointName))
        jointValues.append(motion.frame_joint_channels(10, _aJointName, ['Zrotation', 'Yrotation', 'Xrotation']))
        # print(motion.frame_joint_channels(1, _aJointName, ['Zrotation', 'Yrotation', 'Xrotation']))
    jointValues = np.array(jointValues)
    print(jointValues.shape)
    # print(np.unique(jointValues, axis=0))
    # print(np.unique(jointValues, axis=0).shape)

    
    # Parse file by Bvh_npy(This parser will compute positions of all the joints auto)
    Bvh_parser=Bvh_npy()
    Bvh_parser.parse_file('angry_01_001.bvh')
    all_p, all_r = Bvh_parser.all_frame_poses()
    print([i for i in Bvh_parser.joints])
    print(Bvh_parser.root.children)

    # Traverse the joint tree(Done, and note it in the paper manually!!)
    # print(Bvh_parser.joints['LeftHandIndex1_end'].children)
    # print(Bvh_parser.joints['LThumb_end'].children)
    # print(Bvh_parser.joints['RightHandIndex1_end'].children)
    # print(Bvh_parser.joints['RThumb_end'].children)

    # Draw animation
    XiaKinematicChain=[
        [0, 1, 2, 3, 4, 5], 
        [0, 7, 8, 9, 10, 11], 
        [0, 13, 14, 15, 16, 17, 18], 
        [15, 20, 21, 22, 23, 24], 
        [15, 29, 30, 31, 32, 33]
    ]
    plot_3d_motion(all_p, Bvh_parser.frames, XiaKinematicChain)

    # Iterativly get all the children, and build the skeleton data in tree structure
    # Or other DS will better?
    # Try to imagine the scenario that I will use in the subsequent works. 
    