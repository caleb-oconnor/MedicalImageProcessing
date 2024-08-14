
import copy

import vtk
import numpy as np
import pyvista as pv
import SimpleITK as sitk


class ICPvtk(object):
    def __init__(self, ref, mov, com_matching=True, check_distance=True, select_smallest=True):
        self.com_matching = com_matching
        self.check_distance = check_distance

        self.icp_parameters = {'landmarks': int(np.round(len(mov.points)/10)), 'distance': 1e-5, 'iterations': 1000}

        self.icp = vtk.vtkIterativeClosestPointTransform()
        self.source = None
        self.target = None
        self.reverse_transform = False

        if select_smallest:
            if ref.volume > mov.volume:
                self.source = mov
                self.target = ref
            else:
                self.source = ref
                self.target = mov
                self.reverse_transform = True

    def update_parameters(self, landmarks=None, distance=1e-5, iterations=1000):
        if landmarks:
            self.icp_parameters['landmarks'] = landmarks
        self.icp_parameters['distance'] = distance
        self.icp_parameters['iterations'] = iterations

    def compute_icp(self):
        self.icp.SetSource(self.source)
        self.icp.SetTarget(self.target)
        self.icp.GetLandmarkTransform().SetModeToRigidBody()
        self.icp.SetCheckMeanDistance(1)
        self.icp.SetMeanDistanceModeToRMS()
        self.icp.SetMaximumNumberOfLandmarks(self.icp_parameters['landmarks'])
        self.icp.SetMaximumMeanDistance(self.icp_parameters['distance'])
        self.icp.SetMaximumNumberOfIterations(self.icp_parameters['iterations'])
        self.icp.SetStartByMatchingCentroids(self.com_matching)
        self.icp.Modified()
        self.icp.Update()

        matrix = pv.array_from_vtkmatrix(self.icp.GetMatrix())
        if self.reverse_transform:
            matrix = np.linalg.inv(matrix)

        return matrix

    def compute_error(self):
        matrix = pv.array_from_vtkmatrix(self.icp.GetMatrix())
        new_source = self.source.transform(matrix, inplace=False)

        closest_cells, closest_points = self.target.find_closest_cell(new_source.points, return_closest_point=True)
        d_exact = np.linalg.norm(new_source.points - closest_points, axis=1)
        new_source["distances"] = d_exact
        return {'Min': np.min(d_exact), 'Mean': np.mean(d_exact), 'Max': np.max(d_exact)}


def com_transfer(ref_mesh, mov_mesh):
    ref_com = ref_mesh.center
    mov_com = mov_mesh.center

    rotation_matrix = np.identity(4)
    angles = np.asarray([0, 0, 0])
    translation = mov_com - ref_com
    rotation_matrix[0, 3] = translation[0]
    rotation_matrix[1, 3] = translation[1]
    rotation_matrix[2, 3] = translation[2]

    return rotation_matrix, angles, translation


def transform_mesh(mesh, com, angles, translation, order='xyz'):
    if order == 'xyz':
        mesh_x = mesh.rotate_x(angles[0], point=(com[0], com[1], com[2]))
        mesh_y = mesh_x.rotate_y(angles[1], point=(com[0], com[1], com[2]))
        mesh_z = mesh_y.rotate_z(angles[2], point=(com[0], com[1], com[2]))
        new_mesh = mesh_z.translate((translation[0], translation[1], translation[2]))
    else:
        new_mesh = mesh

    return new_mesh


def euler_transform_with_matrix(matrix):
    transform = sitk.Euler3DTransform()
    transform.SetMatrix(matrix[0:3, 0:3].ravel())
    transform.SetTranslation(matrix[:3, -1])
    transform.ComputeZYXOn()

    return transform


def euler_transform(rotation=None, translation=None, center=None, angles='Radians'):
    if angles == 'Degrees':
        rotation = [rotation[0] * np.pi/180, rotation[1] * np.pi/180, rotation[2] * np.pi/180]

    transform = sitk.Euler3DTransform()
    if rotation is not None:
        transform.SetRotation(rotation[0], rotation[1], rotation[2])
    if translation is not None:
        transform.SetTranslation(translation)
    if center is not None:
        transform.SetCenter(center)

    transform.SetComputeZYX(True)

    return transform


def convert_transformation_matrix_to_angles(matrix):
    r11, r12, r13 = matrix[0][0:3]
    r21, r22, r23 = matrix[1][0:3]
    r31, r32, r33 = matrix[2][0:3]

    angle_z = np.arctan(r21 / r11)
    angle_y = np.arctan(-r31 * np.cos(angle_z) / r11)
    angle_x = np.arctan(r32 / r33)

    return [angle_x * 180 / np.pi, angle_y * 180 / np.pi, angle_z * 180 / np.pi]
