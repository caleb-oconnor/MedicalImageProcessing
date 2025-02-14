"""
Morfeus lab
The University of Texas
MD Anderson Cancer Center
Author - Caleb O'Connor
Email - csoconnor@mdanderson.org

Description:

Structure:

"""

import SimpleITK as sitk


class Transform(object):
    def __init__(self):
        self.transform = None

    def euler(self, rotation=None, translation=None, center=None, angles='Radians'):
        if angles == 'Degrees':
            rotation = [rotation[0] * np.pi/180, rotation[1] * np.pi/180, rotation[2] * np.pi/180]

        self.transform = sitk.Euler3DTransform()
        if rotation is not None:
            self.transform.SetRotation(rotation[0], rotation[1], rotation[2])
        if translation is not None:
            self.transform.SetTranslation(translation)
        if center is not None:
            self.transform.SetCenter(center)

        self.transform.SetComputeZYX(True)

        return self.transform
