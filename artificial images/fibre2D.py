import numpy as np

class Fibre:
    def __init__(self, df, lf, point=(0, 0), point_type='centroid', phiDegxy=0, spcentre=(0, 0)):
        self.df = df
        self.lf = lf
        self.phiDegij = 90 + phiDegxy  # degrees
        self.uvec = self.calc_uvec()
        self.spcentre = np.array(spcentre)
        self.assign_points(point, point_type)
        self.radial_loc()
        self.contain()

    def assign_points(self, point, point_type):
        assert len(point) == 2
        if point_type == 'centroid':
            self.centroid = np.array(point, dtype=int)
            self.start = self.calc_start()
            self.stop = self.calc_stop()
        elif point_type == 'start':
            self.start = np.array(point, dtype=int)
            self.centroid = self.calc_centroid()
            self.stop = self.calc_stop()
        elif point_type == 'stop':
            self.stop = np.array(point, dtype=int)
            self.centroid = self.calc_centroid()
            self.start = self.calc_start()
        else:
            raise ValueError("entry to point_type must be 'start', 'centroid', or 'stop'.")

    def __repr__(self):
        string = ''
        for k, v in vars(self).items():
            string = string + '\n' + k + ': ' + str(v)
        return string

    def calc_uvec(self):
        phi_r = np.deg2rad(self.phiDegij)
        uvec = np.array([np.cos(phi_r), np.sin(phi_r)])
        return uvec

    def calc_start(self):
        val = self.centroid - 0.5 * self.lf * self.uvec
        return val.astype(int)

    def calc_stop(self):
        val = self.centroid + 0.5 * self.lf * self.uvec
        return val.astype(int)

    def calc_centroid(self):
        try:
            centroid = self.start + 0.5 * self.lf * self.uvec
        except AttributeError:
            centroid = self.stop - 0.5 * self.lf * self.uvec
        except AttributeError as err:
            raise err("could not assign centroid.")
        return centroid

    def translate(self, shift=(0, 0)):
        self.centroid = self.centroid + shift
        self.start = self.start + shift
        self.stop = self.stop + shift
        self.radial_loc()

    def radial_loc(self):
        A = self.centroid
        P = self.spcentre
        AP = P - A
        l = np.dot(AP, self.uvec)
        self.nearest = A + l * self.uvec
        self.rad_dist = np.linalg.norm(self.nearest - A)

    def contain(self):
        spcentre = self.spcentre
        start = self.start
        stop = self.stop
        ndim = len(spcentre)
        shift = np.array([0]*ndim)

        for dim in range(ndim):
            coordmax = 2 * spcentre[dim] - 1
            low = min(start[dim], stop[dim])
            high = max(start[dim], stop[dim])
            if low < 0:
                shift[dim] += np.abs(low)
            if high > coordmax:
                shift[dim] += coordmax - high
        if np.linalg.norm(shift) > 0:
            self.translate(shift=shift)