import open3d as o3d
import numpy as np
import time

def main():
    # create visualizer and window.
    vis = o3d.visualization.Visualizer()
    vis.create_window(height=480, width=640)

    # initialize pointcloud instance.
    pcd = o3d.geometry.PointCloud()
    # *optionally* add initial points
    points = np.random.rand(10, 3)
    pcd.points = o3d.utility.Vector3dVector(points)

    # initialize line set
    points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1],
              [0, 1, 1], [1, 1, 1]]
    lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(line_set)

    # include it in the visualizer before non-blocking visualization.
    vis.add_geometry(pcd)


    # to add new points each dt secs.
    dt = 0.05
    # number of points that will be added
    n_new = 10

    previous_t = time.time()

    # run non-blocking visualization. 
    # To exit, press 'q' or click the 'x' of the window.
    keep_running = True
    while keep_running:
        
        if time.time() - previous_t > dt:
            s = time.time()

            # Options (uncomment each to try them out):
            # 1) extend with ndarrays.
            pcd.points.extend(np.random.rand(n_new, 3))

            # 取代舊的points
            # print(np.asarray(pcd.points).shape[0])
            if np.asarray(pcd.points).shape[0] > 10:
                pcd.points = o3d.utility.Vector3dVector(np.random.rand(n_new, 3))
            
            # 2) extend with Vector3dVector instances.
            # pcd.points.extend(
            #     o3d.utility.Vector3dVector(np.random.rand(n_new, 3)))
            
            # 3) other iterables, e.g
            # pcd.points.extend(np.random.rand(n_new, 3).tolist())
            
            vis.update_geometry(pcd)
            previous_t = time.time()
            

        keep_running = vis.poll_events()
        vis.update_renderer()

    vis.destroy_window()

    pass

if __name__=='__main__':
    main()