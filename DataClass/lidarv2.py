# @defgroup Hairoad
#
# copyright Seppe Sels 2024
#
# This code is for internal use only (Uantwerpen, project members)
# Bugs, bugfixes and additions to the code need to be reported to Invilab (contact: Seppe Sels)
# @ingroup Hairoad


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D
from plyfile import PlyData, PlyElement
from scipy.ndimage import distance_transform_edt
import open3d as o3d
import vtk


def load_first_xyz_image(file_path, color_file_path, width, height):
    # Calculate the number of elements
    num_elements = width * height

    # Open the binary file for XYZ data
    with open(file_path, 'rb') as file:
        # Read the required number of bytes
        data = np.fromfile(file, dtype=np.float64, count=num_elements)

    # Reshape the data into the desired dimensions
    image = data.reshape((height, width))

    # Open the binary file for color data
    with open(color_file_path, 'rb') as file:
        # Read the required number of bytes
        color_data = np.fromfile(file, dtype=np.float64, count=num_elements)

    # Reshape the color data into the desired dimensions
    colors = color_data.reshape((height, width))

    return image, colors

def save_as_ply(image, colors, output_file):
    # Extract X, Y, Z coordinates and colors
    vertices = np.array([tuple(row) for row in image], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    colors = np.array([tuple(row) for row in colors], dtype=[('red', 'f4'), ('green', 'f4'), ('blue', 'f4')])

    # Combine vertices and colors
    vertices_with_colors = np.empty(len(vertices), vertices.dtype.descr + colors.dtype.descr)
    for name in vertices.dtype.names:
        vertices_with_colors[name] = vertices[name]
    for name in colors.dtype.names:
        vertices_with_colors[name] = colors[name]

    # Create a PlyElement
    ply_element = PlyElement.describe(vertices_with_colors, 'vertex')

    # Write the PLY file
    PlyData([ply_element]).write(output_file)

def plot_xyz_image(image, colors):
    # Extract X, Y, Z coordinates
    X = image[:, 0]
    Y = image[:, 1]
    Z = image[:, 2]

    # Extract RGB colors
    R = colors[:, 0]
    G = colors[:, 1]
    B = colors[:, 2]
    rgb = np.vstack((R, G, B)).T / 255.0

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, c=rgb, marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()



def plot_xyz_with_vtk(XYZ, colors):
    # Create a vtkPoints object and insert the points into it
    points = vtk.vtkPoints()
    for i in range(XYZ.shape[0]):
        points.InsertNextPoint(XYZ[i, :])

    # Create a vtkPolyData object and set the points
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)

    # Create a vtkUnsignedCharArray object and set the colors
    vtk_colors = vtk.vtkUnsignedCharArray()
    vtk_colors.SetNumberOfComponents(3)
    vtk_colors.SetName("Colors")
    for i in range(colors.shape[0]):
        vtk_colors.InsertNextTuple3(colors[i, 0], colors[i, 1], colors[i, 2])

    # Set the colors to the polydata
    polydata.GetPointData().SetScalars(vtk_colors)

    # Create a vtkVertexGlyphFilter object and set the input data
    vertex_filter = vtk.vtkVertexGlyphFilter()
    vertex_filter.SetInputData(polydata)
    vertex_filter.Update()

    # Create a vtkPolyDataMapper object and set the input connection
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(vertex_filter.GetOutputPort())

    # Create a vtkActor object and set the mapper
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # Create a vtkRenderer object and add the actor
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.2, 0.4)

    # Create a vtkRenderWindow object and add the renderer
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)

    # Create a vtkRenderWindowInteractor object and set the render window
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Start the interaction
    render_window.Render()
    render_window_interactor.Start()



def plot_xyz_with_vtk_and_get_depth(XYZ, colors, width=1000, height=1000, point_size=3):
    # Create a vtkPoints object and insert the points into it
    points = vtk.vtkPoints()
    for i in range(XYZ.shape[0]):
        points.InsertNextPoint(XYZ[i, :])

    # Create a vtkPolyData object and set the points
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)

    # Create a vtkUnsignedCharArray object and set the colors
    vtk_colors = vtk.vtkUnsignedCharArray()
    vtk_colors.SetNumberOfComponents(3)
    vtk_colors.SetName("Colors")
    for i in range(colors.shape[0]):
        vtk_colors.InsertNextTuple3(colors[i, 0], colors[i, 1], colors[i, 2])

    # Set the colors to the polydata
    polydata.GetPointData().SetScalars(vtk_colors)

    # Create a vtkVertexGlyphFilter object and set the input data
    vertex_filter = vtk.vtkVertexGlyphFilter()
    vertex_filter.SetInputData(polydata)
    vertex_filter.Update()

    # Create a vtkPolyDataMapper object and set the input connection
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(vertex_filter.GetOutputPort())

    # Create a vtkActor object and set the mapper
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(point_size)  # Set the point size

    # Create a vtkRenderer object and add the actor
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.2, 0.4)

    # Create a vtkRenderWindow object and add the renderer
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(width, height)

    # Create a vtkRenderWindowInteractor object and set the render window
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Render the scene
    render_window.Render()

    # Capture the depth buffer
    z_buffer = vtk.vtkFloatArray()
    render_window.GetZbufferData(0, 0, width - 1, height - 1, z_buffer)

    # Convert the depth buffer to a numpy array
    depth_image = np.zeros((height, width), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            depth_image[i, j] = z_buffer.GetValue(i * width + j)

    # Start the interaction
    render_window_interactor.Start()

    return depth_image



def detect_and_plot_largest_plane(XYZ):
    # Convert the numpy array to an Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(XYZ)

    # Detect the largest plane using RANSAC
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                             ransac_n=3,
                                             num_iterations=1000)
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)

    # Convert the inlier and outlier point clouds back to numpy arrays
    inlier_XYZ = np.asarray(inlier_cloud.points)
    outlier_XYZ = np.asarray(outlier_cloud.points)

    # Plot the original point cloud with the plane points highlighted
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(outlier_XYZ[:, 0], outlier_XYZ[:, 1], outlier_XYZ[:, 2], c='b', marker='o', label='Outliers')
    ax.scatter(inlier_XYZ[:, 0], inlier_XYZ[:, 1], inlier_XYZ[:, 2], c='r', marker='o', label='Plane')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.show()

    [a, b, c, d] = plane_model

    # Extract the normal of the plane
    normal = np.array([a, b, c])

    # Calculate the rotation matrix to align the plane with the XY axis
    z_axis = np.array([0, 1, 0])
    rotation_axis = np.cross(normal, z_axis)
    rotation_angle = np.arccos(np.dot(normal, z_axis) / (np.linalg.norm(normal) * np.linalg.norm(z_axis)))
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)

    # Apply the rotation to the point cloud
    pcd.rotate(rotation_matrix)
    # Convert the transformed point cloud back to a numpy array
    transformed_XYZ = np.asarray(pcd.points)
    transformed_XYZ= transformed_XYZ[:, [0, 2, 1]] # something went wrong in the transformation code. lazy fix: Todo: fix the transformation code
    return transformed_XYZ



def fill_depth_image(depth_image):
    # Create a mask for pixels with value larger than 0.9999 or smaller than 0.00001
    mask = (depth_image > 0.9999) | (depth_image < 0.00001)
    # Compute the distance transform
    distance, indices = distance_transform_edt(mask, return_indices=True)

    # Iterate through each pixel in the depth image
    for i in range(depth_image.shape[0]):
        for j in range(depth_image.shape[1]):
            if depth_image[i, j] > 0.9999 or depth_image[i, j] < 0.00001:
                # Get the nearest neighbor's coordinates
                ni, nj = indices[:, i, j]
                if distance[i, j] <= 10 :
                    # Fill the current pixel with the nearest neighbor's value
                    depth_image[i, j] = depth_image[ni, nj]
                else:
                    # Set the current pixel to NaN
                    depth_image[i, j] = np.nan

    return depth_image



if __name__ == "__main__":
    # Example usage
    file_path = r"\\datanasop3mech\ProjectData\4_Other\Hairoad\Experiments\November2024\2024-11-28_10-48-44\test_2024_11_28__10_48_44_698021.bin"
    color_file_path = r"\\datanasop3mech\ProjectData\4_Other\Hairoad\Experiments\November2024\2024-11-28_10-48-44\test_2024_11_28__10_48_44_698021_colors.bin"
    output_file = 'output.ply'

    width = 3
    height = 8857
    image, colors = load_first_xyz_image(file_path, color_file_path, width, height)
    print(image)
    save_as_ply(image, colors, output_file)

    XYZ, colors = load_first_xyz_image(file_path, color_file_path, width, height)
    #plot_xyz_with_vtk(XYZ, colors)

    XYZ_transformed = detect_and_plot_largest_plane(XYZ)

    depth_image = plot_xyz_with_vtk_and_get_depth(XYZ_transformed, colors) # note, not done with vtk but with open3D
    # plot the depth image with a title
    plt.title('Depth Image')
    plt.imshow(depth_image)
    plt.show()
    depth_image = fill_depth_image(depth_image) # a verry simple depth filling algorithm. Todo: implement a better one
    plt.title('Depth Image, filled')
    plt.imshow(depth_image, cmap='viridis', vmin=0.55, vmax=0.7)

    plt.colorbar()
    plt.show()
    # add a histogram slider
