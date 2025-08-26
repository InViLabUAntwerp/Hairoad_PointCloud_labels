#!/usr/bin/env python3
import numpy as np


def write_bundler_file(T, K, output_file):
    """
    Write a Bundler file (v0.3) for MeshLab texturing.

    Parameters:
      T : (4,4) np.ndarray
          The provided transformation matrix (assumed camera-to-world).
      K : (3,3) np.ndarray
          The intrinsic matrix.
      output_file : str
          The filename for the output Bundler file.
    """
    # Compute the average focal length from the intrinsics.
    # Bundler expects a single focal length (in pixels).
    f = (K[0, 0] + K[1, 1]) / 2.0

    # If no radial distortion is provided, set them to zero.
    k1, k2 = 0.0, 0.0

    # Extract the rotation matrix (R) and translation vector (t) from T.


    # For Bundler, we need the world-to-camera transformation.
    # If T is camera-to-world, its inverse is:
    #   R_inv = R^T, and t_inv = -R^T * t
    T_inv = np.linalg.inv(T)
    R_inv = T_inv[:3, :3]
    t_inv = T_inv[:3, 3]
    # Build the Bundler file content.
    lines = []
    lines.append("# Bundle file v0.3")
    lines.append("1 0")  # one camera, zero points
    lines.append(f"{f:.8f} {k1:.8f} {k2:.8f}")

    # Add the rotation matrix rows (world-to-camera rotation).
    for row in R_inv:
        lines.append(" ".join(f"{val:.8f}" for val in row))
    # Add the translation vector.
    lines.append(" ".join(f"{val:.8f}" for val in t_inv))

    bundler_content = "\n".join(lines)

    # Write the content to the specified output file.
    with open(output_file, "w") as f_out:
        f_out.write(bundler_content)

    print(f"Bundler file written to {output_file}")


if __name__ == "__main__":
    # Define your provided transformation matrix (T) and intrinsic matrix (K).
    T = np.array([
        [1.937702560325514911e-01, -9.768608322764555929e-01, 9.053177476019336201e-02, -2.933615975633154238e-01],
        [4.793064133735732146e-01, 1.374820000771576067e-02, -8.775399415955432714e-01, -2.613615639512909539e-01],
        [8.559897487563978835e-01, 2.134335994184110019e-01, 4.708796541190335416e-01, 1.810074541814694626e-01],
        [0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]
    ])

    K = np.array([
        [1.770689941406250000e+03, 0.000000000000000000e+00, 6.852999877929687500e+02],
        [0.000000000000000000e+00, 1.765030029296875000e+03, 4.927000122070312500e+02],
        [0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]
    ])

    # Output file for the Bundler file.
    output_filename = "camera3.out"

    # Write the bundler file (with the inverted transformation matrix).
    write_bundler_file(T, K, output_filename)