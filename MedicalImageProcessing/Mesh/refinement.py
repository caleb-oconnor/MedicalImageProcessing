"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:

Structure:

"""

import vtk
import pyacvd
import pyvista as pv


class Refinement(object):
    def __init__(self, mesh):
        self.mesh = mesh

    def smooth(self, iterations=20, angle=60, passband=0.001):
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputData(self.mesh)
        smoother.SetNumberOfIterations(iterations)
        smoother.BoundarySmoothingOff()
        smoother.FeatureEdgeSmoothingOff()
        smoother.SetFeatureAngle(angle)
        smoother.SetPassBand(passband)
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOff()
        smoother.Update()
        self.mesh = pv.PolyData(smoother.GetOutput())

        return self.mesh

    def cluster(self, points=None):
        if points is None:
            points = self.compute_points()
        clus = pyacvd.Clustering(self.mesh)
        clus.cluster(points)
        self.mesh = clus.create_mesh()

        return self.mesh

    def decimate(self, percent=None):
        if percent is None:
            percent = self.compute_point_percentage()

        self.mesh.decimate(percent)

        return self.mesh

    def compute_points(self):
        return np.round(10 * np.sqrt(self.mesh.number_of_points))

    def compute_point_percentage(self):
        points = self.compute_points()

        return 1 - (points / self.mesh.number_of_points)
