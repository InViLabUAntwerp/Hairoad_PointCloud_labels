import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D
from plyfile import PlyData, PlyElement


def save_as_ply(image, output_file):
    # Extract X, Y, Z coordinates
    vertices = np.array([tuple(row) for row in image], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    # Create a PlyElement
    ply_element = PlyElement.describe(vertices, 'vertex')

    # Write the PLY file
    PlyData([ply_element]).write(output_file)
def load_first_xyz_image(file_path, width, height):
    # Calculate the number of elements
    num_elements = width * height

    # Open the binary file
    with open(file_path, 'rb') as file:
        # Read the required number of bytes
        data = np.fromfile(file, dtype=np.float64, count=num_elements)

    # Reshape the data into the desired dimensions
    image = data.reshape((height, width))

    return image
def plot_xyz_image(image):
    # Extract X, Y, Z coordinates
    X = image[:, 0]
    Y = image[:, 1]
    Z = image[:, 2]

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, c=Z, cmap='viridis', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

if __name__ == "__main__":
    # Example usage
    file_path = r"\\datanasop3mech\ProjectData\4_Other\Hairoad\Experiments\November2024\2024-11-28_10-48-44\test_2024_11_28__10_48_44_698021.bin"
    output_file = 'output.ply'

    width = 3
    height = 8857
    image = load_first_xyz_image(file_path, width, height)
    print(image)
    save_as_ply(image, output_file)
    plot_xyz_image(image)
