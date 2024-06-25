from ilastik.plugins_default import objfeat_spherical_projection_base


class SphericalProjection3D(objfeat_spherical_projection_base.SphericalProjection):
    def __init__(self):
        super().__init__()
        self.ndim = 3
