###############################################################################
#   ilastik: interactive learning and segmentation toolkit
#
#       Copyright (C) 2011-2014, the ilastik developers
#                                <team@ilastik.org>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# In addition, as a special exception, the copyright holders of
# ilastik give you permission to combine ilastik with applets,
# workflows and plugins which are not covered under the GNU
# General Public License.
#
# See the LICENSE file for details. License information is also available
# on the ilastik web site at:
# 		   http://ilastik.org/license.html

# Plugin written by Oane Gros.
# Does feature extraction by spherical projection texture
# Does a user-settable projection along rays from the centroid in gauss-legendre quadrature
# Mapping the data to a 2D sphericall surface
# This is decomposed into a spherical harmonics power spectrum
# The resulting feature is an undersampled (for feature reduction) spectrum
###############################################################################

# core ilastik
from ilastik.plugins import ObjectFeaturesPlugin
from ilastik.plugins_default.convex_hull_feature_description import fill_feature_description
import ilastik.applets.objectExtraction.opObjectExtraction

# core
import vigra
import logging
import numpy as np

# transformations and speedup
from numba import jit, typeof, typed, types
from skimage.transform import resize
from skimage.filters import gaussian
from skimage import img_as_bool
import pyshtools as pysh
from pyshtools.shtools import GLQGridCoord
from pyshtools.expand import SHExpandGLQ
from pyshtools.spectralanalysis import spectrum


# saving/loading LUT
import pickle as pickle
from pathlib import Path

# temp
import time
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class SphericalProjection(ObjectFeaturesPlugin):
    ndim = None
    margin = 0  # necessary for calling compute_local

    # projection order is
    projectionorder = ["MAX projection", "MIN projection", "SHAPE projection", "MEAN projection"]
    # It might be more maintainable to have this as a list of tuples with the axis in there, but you would probably need to do string comparison instead of boolean checks in the lookup
    # projections = [("MAX projection",-1), ("MIN projection",-1), ("Length",-1), ("MEAN projection",-1)]
    # or best would be a dictionary of featname : func, but this might give numba-issues
    projections = np.zeros(len(projectionorder), dtype=bool)  # contains selection of which projections should be done

    # Set in compute_local on first run:
    raysLUT = None
    fineness = None
    scale = 0

    def availableFeatures(self, image, labels):

        if labels.ndim == 3:
            names = [  # compute_local actually uses the last number to set self.scale, so be careful changing
                "resolution 10x10x10",
                "resolution 20x20x20",
                "resolution 40x40x40",
                "resolution 80x80x80",
                # "resolution 160x160x160",
            ]
            for proj in self.projectionorder:
                names.append(proj)
            tooltips = {}
            result = dict((n, {}) for n in names)
            result = self.fill_properties(result)
            for f, v in result.items():
                v["tooltip"] = "Spherical projections " + f
        else:
            result = {}

        return result

    @staticmethod
    def fill_properties(features):
        # fill in the detailed zinformation about the features.
        # features should be a dict with the feature_name as key.
        # NOTE, this function needs to be updated every time skeleton features change
        for feature in features:
            features[feature]["displaytext"] = feature
            if "resolution" in feature:
                features[feature]["group"] = "Max resolution"
                features[feature][
                    "detailtext"
                ] = "Object is rescaled to max checked resolution before spherical projection, lower is faster."
            if "projection" in feature:
                features[feature]["group"] = "Projection types"
                features[feature][
                    "detailtext"
                ] = "All checked projections currently provide int(1D_resolution*π) features"
            features[feature]["margin"] = 0  # needs to be set to trigger compute_local
        return features

    def unwrap_and_expand(self, image, binary_bbox, axes, features):

        t0 = time.time()
        rawbbox = image
        mask_object = binary_bbox

        if self.raysLUT == None:
            self.raysLUT = self.get_ray_table()

        cube = resize(image, (self.scale, self.scale, self.scale), preserve_range=True)
        mask_cube = resize(img_as_bool(mask_object), (self.scale, self.scale, self.scale), order=0)
        segmented_cube = np.where(mask_cube, cube, -1)
        t1 = time.time()
        unwrapped = lookup_spherical(segmented_cube, self.raysLUT, self.fineness, self.projections)

        t2 = time.time()

        result = {}
        projectedix = 0
        for which_proj, projected in enumerate(self.projections):
            if projected:
                projection = unwrapped[:, :, projectedix].astype(float)
                # plt.imsave('/Users/oanegros/Documents/screenshots/tmp_unwrapped2/'+ str(t0) + "_" + str(np.count_nonzero(mask_object))+ "_" + str(np.count_nonzero(mask_object==0))+self.projectionorder[which_proj]+"unwrapGLQ_masked.png",  projection)
                projectedix += 1  #
                zero, w = pysh.expand.SHGLQ(self.fineness)
                coeffs = pysh.expand.SHExpandGLQ(projection, w=w, zero=zero)
                power_per_dlogl = spectrum(coeffs, unit="per_dlogl", base=2)
                # print(time.time()-t4)

                # bin in 2log spaced bins
                bins = np.logspace(0, np.log2(len(power_per_dlogl)), num=20, base=2, endpoint=True)
                bin_ix, current_bin, means = 0, [], []
                for degree, power in enumerate(power_per_dlogl[1:]):
                    current_bin.append(power)
                    if degree + 1 >= bins[bin_ix]:
                        if len(current_bin) > 0:  # this is for high bins/indices
                            means.append(np.mean(current_bin))
                        bin_ix += 1
                        current_bin = []
                result[self.projectionorder[which_proj]] = np.log2(means)
                # result[self.projectionorder[which_proj]] = np.log2(power_per_dlogl)

        t3 = time.time()
        print("time to do full unwrap and expand: \t", t3 - t0)
        return result

    def _do_3d(self, image, binary_bbox, features, axes):
        results = []
        features = list(features.keys())
        results.append(self.unwrap_and_expand(image, binary_bbox, axes, features))
        return results[0]

    def init_selection(self, features):
        for featurename in features:
            if "resolution" in featurename:
                self.scale = max(self.scale, int(featurename.split(" ")[-1].split("x")[-1]))
                self.fineness = int(np.pi * self.scale)
            else:
                for ix, proj in enumerate(self.projectionorder):
                    if proj == featurename:
                        self.projections[ix] = True
        return

    def compute_local(self, image, binary_bbox, features, axes):
        if self.fineness == None:
            self.init_selection(features)
        # print(binary_bbox.shape)
        orig_bbox = binary_bbox
        margin = [(np.min(dim), np.max(dim) + 1) for dim in np.nonzero(binary_bbox)]
        image = image[margin[0][0] : margin[0][1], margin[1][0] : margin[1][1], margin[2][0] : margin[2][1]]
        binary_bbox = binary_bbox[margin[0][0] : margin[0][1], margin[1][0] : margin[1][1], margin[2][0] : margin[2][1]]

        assert np.sum(orig_bbox) - np.sum(binary_bbox) == 0
        return self.do_channels(self._do_3d, image, binary_bbox=binary_bbox, features=features, axes=axes)

    def save_ray_table(self, rays):
        # save a pickle of the rayLUT, requires retyping of the dictionary
        # this is because the rays are ragged, and not single-length
        outLUT = {}  # need to un-type the dictionary for pickling
        for k, v in rays.items():
            outLUT[k] = v
        name = "sphericalLUT" + str(self.scale) + ".pickle"
        with open(Path(__file__).parent / name, "wb") as ofs:
            pickle.dump(outLUT, ofs, protocol=pickle.HIGHEST_PROTOCOL)
        return

    def get_ray_table(self):
        # try to load or generate new
        # loading requires retyping of the dictionary for numba, which slows it down
        try:
            t0 = time.time()
            name = "sphericalLUT" + str(self.scale) + ".pickle"
            with open(Path(__file__).parent / name, "rb") as handle:
                newLUT = pickle.load(handle)
            typed_rays = typed.Dict.empty(
                key_type=typeof((1, 1)),
                value_type=typeof(np.zeros((1, 3), dtype=np.int16)),
            )
            for k, v in newLUT.items():
                typed_rays[k] = v
            print("loaded ray table in: ", time.time() - t0)
            return typed_rays
        except Exception as e:
            return self.generate_ray_table()

    def generate_ray_table(self):
        # make new ray table; here do all tasks outside of numba handling (end of file)
        print("recalculating LUT")
        t0 = time.time()
        rays = typed.Dict.empty(
            key_type=typeof((1, 1)),
            value_type=typeof(np.zeros((1, 3), dtype=np.int16)),
        )
        fill_ray_table(self.scale, GLQGridCoord(self.fineness), rays)
        self.save_ray_table(rays)
        t1 = time.time()
        print("time to make ray table: ", t1 - t0)
        return rays


# All numba-accelerated functions cannot receive self, so are not class functions


@jit(nopython=True)
def lookup_spherical(img, raysLUT, fineness, projections):
    unwrapped = np.zeros((fineness + 1, fineness * 2 + 1, np.sum(projections)), dtype=np.float64)
    for loc, ray in raysLUT.items():
        # TODO update indexing to "values = img[ray[:,0],ray[:,1],ray[:,2]]" or similar once numba multiple advanced indexing is merged https://github.com/numba/numba/pull/8491
        values = np.zeros(ray.shape[0])
        for ix, voxel in enumerate(ray):
            values[ix] = img[voxel[0], voxel[1], voxel[2]]

            if values[ix] < 0:
                # quit when outside of object mask -  all outside of mask are set to -1
                if ix != 0:  # centroid is not outside of mask
                    values = values[:ix]
                    break
        proj = 0
        ray = ray.astype(np.float64)
        if projections[0]:  # MAX
            unwrapped[loc[1], loc[0], proj] = np.amax(values)
            proj += 1
        if projections[1]:  # MIN
            unwrapped[loc[1], loc[0], proj] = np.amin(values)
            proj += 1
        if projections[2]:  # SHAPE
            unwrapped[loc[1], loc[0], proj] = len(values)
            proj += 1
        if projections[3]:  # MEAN
            unwrapped[loc[1], loc[0], proj] = np.mean(values)
            proj += 1
    return unwrapped


# ---- only used in generating LUT ----


@jit(nopython=True)
def fill_ray_table(scale, GLQcoords, rays):
    centroid = (
        np.array([scale, scale, scale], dtype=np.float32) / 2.0
    ) + 0.1  # slightly offset to reduce float rounding errors at the border
    glq_lat, glq_lon = np.deg2rad(GLQcoords[0]), np.deg2rad(GLQcoords[1])

    for phi_ix, lon in enumerate(glq_lon):
        for theta_ix, lat in enumerate(glq_lat):
            ray = np.array(
                [
                    np.sin((np.pi / 2) - lat) * np.cos(lon),
                    np.sin((np.pi / 2) - lat) * np.sin(lon),
                    np.cos((np.pi / 2) - lat),
                ]
            )
            pixels = march(ray, centroid, scale, marchlen=0.003).astype(np.int16).copy()

            # find unique
            diff = np.sum(pixels[1:, :] - pixels[:-1, :], axis=1)
            nz = np.nonzero(diff)[0]
            unique_pixels = np.zeros((nz.shape[0], 3), dtype=np.int16)
            unique_pixels[0, :] = pixels[0]
            for ix, val in enumerate(nz):  # ix+1 this is because np.insert doesnt njit
                unique_pixels[ix + 1, :] = pixels[val + 1]

            rays[(phi_ix, theta_ix)] = unique_pixels
    return rays


@jit(nopython=True)
def march(ray, centroid, scale, marchlen):
    increment = ray * marchlen
    distances = []
    normals = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
    bounds = [np.array([scale, scale, scale]).astype(np.float64), np.array([0.0, 0.0, 0.0])]
    for normal in normals:
        for bound in bounds:
            intersect = isect_dist_line_plane(centroid, ray, bound, normal)
            distances.append(intersect)
    est_length = min(distances) / marchlen
    end = est_length * increment + centroid
    pixels = (
        np.linspace(centroid[0], end[0], int(est_length)),
        np.linspace(centroid[1], end[1], int(est_length)),
        np.linspace(centroid[2], end[2], int(est_length)),
    )
    pixels = np.stack(pixels).T
    return pixels


# intersection function edited from https://stackoverflow.com/questions/5666222/3d-line-plane-intersection
@jit(nopython=True)
def isect_dist_line_plane(centroid, raydir, planepoint, planenormal, epsilon=1e-6):
    dot = np.dot(planenormal, raydir)
    if np.abs(dot) > epsilon:
        w = centroid - planepoint
        fac = -np.dot(planenormal, w) / dot
        if fac > 0:
            return fac
    return np.inf
