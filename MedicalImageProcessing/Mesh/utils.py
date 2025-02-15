"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:

Structure:

"""

import numpy as np
import pyvista as pv


def merge(mesh_list):
    mesh = pv.merge(mesh_list)

    return mesh


def rotation_matrix_to_angles(matrix):
    """
    Gets the angles for a rotation matrix in zyx order.

    :param matrix: rotation matrix
    :return:
    """
    r11, r12, r13 = matrix[0][0:3]
    r21, r22, r23 = matrix[1][0:3]
    r31, r32, r33 = matrix[2][0:3]

    angle_y = np.arcsin(r13) * (180 / np.pi)
    if np.abs(r13) < .999:
        angle_x = np.arctan(-r23 / r33) * (180 / np.pi)
        angle_z = np.arctan(-r12 / r11) * (180 / np.pi)
    else:
        angle_x = np.arctan(r32 / r22) * (180 / np.pi)
        angle_z = 0

    return [angle_z, angle_y, angle_x]
