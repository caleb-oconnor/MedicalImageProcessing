
import copy

import vtk
import numpy as np
import pyvista as pv
import SimpleITK as sitk

# from scipy.spatial.transform import Rotation as R


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


def transform_mesh(mesh, com, angles, translation, order='xyz'):
    if order == 'xyz':
        mesh_x = mesh.rotate_x(angles[0], point=(com[0], com[1], com[2]))
        mesh_y = mesh_x.rotate_y(angles[1], point=(com[0], com[1], com[2]))
        mesh_z = mesh_y.rotate_z(angles[2], point=(com[0], com[1], com[2]))
        new_mesh = mesh_z.translate((translation[0], translation[1], translation[2]))
    else:
        new_mesh = mesh

    return new_mesh


def euler_transform(rotation=None, translation=None, center=None, zyx=True, angles='Radians'):
    if angles == 'Degrees':
        rotation = [rotation[0] * np.pi/180, rotation[1] * np.pi/180, rotation[2] * np.pi/180]

    transform = sitk.Euler3DTransform()
    if rotation is not None:
        transform.SetRotation(rotation[0], rotation[1], rotation[2])
    if translation is not None:
        transform.SetTranslation(translation)
    if center is not None:
        transform.SetCenter(center)
    # if zyx:
    #     transform.SetComputeXYZ(True)

    return transform


def convert_transformation_matrix_to_angles(matrix):
    r11, r12, r13 = matrix[0][0:3]
    r21, r22, r23 = matrix[1][0:3]
    r31, r32, r33 = matrix[2][0:3]

    angle_z = np.arctan(r21 / r11)
    angle_y = np.arctan(-r31 * np.cos(angle_z) / r11)
    angle_x = np.arctan(r32 / r33)

    return [angle_x * 180 / np.pi, angle_y * 180 / np.pi, angle_z * 180 / np.pi]


# class Open3dRegistration(object):
#     from open3d.geometry import PointCloud
#     from open3d.utility import Vector3dVector
#     from open3d.pipelines.registration import registration_icp, TransformationEstimationPointToPoint, \
#         ICPConvergenceCriteria
#     def __init__(self, ref_pv=None, mov_pv=None, ref_pcd=None, mov_pcd=None):
#         self.ref_pv = ref_pv
#         self.mov_pv = mov_pv
#         self.ref_pcd = ref_pcd
#         self.mov_pcd = mov_pcd
#
#         self.icp_settings = {'max_distance': 1, 'max_iteration': 1000, 'rmse': 0.0000001, 'fitness': 0.0000001}
#
#         self.ref_com = None
#         self.mov_com = None
#         self.initial_transform = None
#         self.compute_com()
#
#         self.reg = None
#         self.parameters = {'transform': None, 'fitness': None, 'rmse': None}
#         self.angles = None
#         self.translation = None
#         self.new_mesh = None
#
#     def compute_com(self):
#         if not self.ref_pcd:
#             if self.ref_pv:
#                 self.ref_pcd = PointCloud()
#                 self.ref_pcd.points = Vector3dVector(self.ref_pv.points)
#                 self.ref_com = self.ref_pcd.get_center()
#         else:
#             self.ref_com = self.ref_pcd.get_center()
#
#         if not self.mov_pcd:
#             if self.mov_pv:
#                 self.mov_pcd = PointCloud()
#                 self.mov_pcd.points = Vector3dVector(self.mov_pv.points)
#                 self.mov_com = self.mov_pcd.get_center()
#         else:
#             self.mov_com = self.mov_pcd.get_center()
#
#         c = self.mov_com - self.ref_com
#         self.initial_transform = np.asarray([[1, 0, 0, c[0]], [0, 1, 0, c[1]], [0, 0, 1, c[2]], [0, 0, 0, 1]])
#
#     def set_initial_transform(self, transform):
#         self.initial_transform = transform
#
#     def set_icp_settings(self, max_distance=None, max_iteration=None, rmse=None, fitness=None):
#         if max_distance is not None:
#             self.icp_settings['max_distance'] = max_distance
#
#         if max_iteration is not None:
#             self.icp_settings['max_iteration'] = max_iteration
#
#         if rmse is not None:
#             self.icp_settings['rmse'] = rmse
#
#         if fitness is not None:
#             self.icp_settings['fitness'] = fitness
#
#     def compute(self):
#         self.reg = registration_icp(self.ref_pcd, self.mov_pcd, self.icp_settings['max_distance'],
#                                     self.initial_transform, TransformationEstimationPointToPoint(),
#                                     ICPConvergenceCriteria(max_iteration=self.icp_settings['max_iteration'],
#                                                            relative_rmse=self.icp_settings['rmse'],
#                                                            relative_fitness=self.icp_settings['fitness']))
#
#         self.parameters['transform'] = self.reg.transformation
#         self.parameters['fitness'] = self.reg.fitness
#         self.parameters['rmse'] = self.reg.inlier_rmse
#
#         r11, r12, r13 = self.parameters['transform'][0][0:3]
#         r21, r22, r23 = self.parameters['transform'][1][0:3]
#         r31, r32, r33 = self.parameters['transform'][2][0:3]
#
#         angle_z = np.arctan(r21 / r11)
#         angle_y = np.arctan(-r31 * np.cos(angle_z) / r11)
#         angle_x = np.arctan(r32 / r33)
#
#         self.new_mesh = self.ref_pcd.transform(self.parameters['transform'])
#         self.angles = [angle_x * 180 / np.pi, angle_y * 180 / np.pi, angle_z * 180 / np.pi]
#         self.translation = self.new_mesh.get_center() - self.ref_com
