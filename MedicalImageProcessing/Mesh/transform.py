"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:

Structure:

"""

import copy
import pyvista as pv


class Transform(object):
    def __init__(self, mesh):
        self.mesh = mesh
        self.transformed_mesh = copy.deepcopy(self.mesh)

    def matrix(self, matrix):
        transform = pv.Transform()
        transform.matrix = matrix

        self.transformed_mesh.transform(transform, inplace=True)

        return self.transformed_mesh

    def angles_only(self, com, angles, order='xyz'):
        for o in order:
            if o == 'x':
                self.transformed_mesh.rotate_x(angles[0], point=(com[0], com[1], com[2]), inplace=True)

            if o == 'y':
                self.transformed_mesh.rotate_y(angles[0], point=(com[0], com[1], com[2]), inplace=True)

            if o == 'z':
                self.transformed_mesh.rotate_z(angles[0], point=(com[0], com[1], com[2]), inplace=True)

        return self.transformed_mesh

    def translation_only(self, translation):
        self.transformed_mesh.translate((translation[0], translation[1], translation[2]), inplace=True)

        return self.transformed_mesh
