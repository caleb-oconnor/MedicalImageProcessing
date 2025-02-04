
import copy

import vtk
import numpy as np
import pyvista as pv
import SimpleITK as sitk

from open3d.geometry import PointCloud
from open3d.utility import Vector3dVector
from open3d.pipelines.registration import registration_icp, TransformationEstimationPointToPoint, ICPConvergenceCriteria


class ICPvtk(object):
    def __init__(self, ref, mov):
        self.source = mov
        self.target = ref

        self.landmarks = int(np.round(len(mov.points)/10)),
        self.distance = 1e-5
        self.iterations = 1000

        self.icp = vtk.vtkIterativeClosestPointTransform()

        self.reverse_transform = False

    def update_parameters(self, landmarks=None, distance=None, iterations=None):
        if landmarks:
            self.landmarks = landmarks

        if distance:
            self.distance = distance

        if iterations:
            self.iterations = iterations

    def compute_icp(self, com_matching=True):
        self.icp.SetSource(self.source)
        self.icp.SetTarget(self.target)
        self.icp.GetLandmarkTransform().SetModeToRigidBody()
        self.icp.SetCheckMeanDistance(1)
        self.icp.SetMeanDistanceModeToRMS()
        self.icp.SetMaximumNumberOfLandmarks(self.landmarks)
        self.icp.SetMaximumMeanDistance(self.distance)
        self.icp.SetMaximumNumberOfIterations(self.iterations)
        self.icp.SetStartByMatchingCentroids(com_matching)
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


class ICPo3d(object):
    def __init__(self, ref_pv=None, mov_pv=None):
        self.ref_pcd = PointCloud()
        self.ref_pcd.points = Vector3dVector(ref_pv.points)
        self.mov_pcd = PointCloud()
        self.mov_pcd.points = Vector3dVector(mov_pv.points)

        self.distance = 1
        self.iterations = 1000
        self.rmse = 1e-7
        self.fitness = 1e-7

        self.initial_transform = None
        self.compute_com()

        self.registration = None
        self.parameter_results = {'transform': None, 'fitness': None, 'rmse': None}
        self.angles = None
        self.translation = None
        self.new_mesh = None

    def set_initial_transform(self, transform):
        self.initial_transform = transform

    def set_icp_settings(self, distance=None, iterations=None, rmse=None, fitness=None):
        if max_distancee:
            self.distance = distance

        if max_iteration:
            self.iterations = iterations

        if rmse:
            self.rmse = rmse

        if fitness:
            self.fitness = fitness

    def compute(self, com_matching=True):
        if com_matching:
            c = self.mov_pcd.get_center() - self.ref_pcd.get_center()
            self.initial_transform = np.asarray([[1, 0, 0, c[0]], [0, 1, 0, c[1]], [0, 0, 1, c[2]], [0, 0, 0, 1]])
        else:
            self.initial_transform = np.identity(4, dtye=np.float32)

        self.registration = registration_icp(self.ref_pcd, self.mov_pcd, self.distance, self.initial_transform,
                                             TransformationEstimationPointToPoint(),
                                             ICPConvergenceCriteria(max_iteration=self.iterations,
                                                                    relative_rmse=self.rmse,
                                                                    relative_fitness=self.fitness))

        self.parameter_results['transform'] = self.registration.transformation
        self.parameter_results['fitness'] = self.registration.fitness
        self.parameter_results['rmse'] = self.registration.inlier_rmse

        r11, r12, r13 = self.parameter_results['transform'][0][0:3]
        r21, r22, r23 = self.parameter_results['transform'][1][0:3]
        r31, r32, r33 = self.parameter_results['transform'][2][0:3]

        angle_z = np.arctan(r21 / r11)
        angle_y = np.arctan(-r31 * np.cos(angle_z) / r11)
        angle_x = np.arctan(r32 / r33)

        self.new_mesh = self.ref_pcd.transform(self.parameter_results['transform'])
        self.angles = [angle_x * 180 / np.pi, angle_y * 180 / np.pi, angle_z * 180 / np.pi]
        self.translation = self.new_mesh.get_center() - self.ref_com

    def correspondence_array(self):
        return self.registration.correspondence_set


def com_transfer(ref_mesh, mov_mesh):
    ref_com = ref_mesh.center
    mov_com = mov_mesh.center

    rotation_matrix = np.identity(4)
    angles = np.asarray([0, 0, 0])
    translation = mov_com - ref_com
    rotation_matrix[0, 3] = -translation[0]
    rotation_matrix[1, 3] = -translation[1]
    rotation_matrix[2, 3] = -translation[2]

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
    """
    Gets the angles for a rotation matrix in zyx order. Angles transform the moving to the reference.

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

