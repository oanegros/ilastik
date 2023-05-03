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

import threading
import tifffile

# saving/loading LUT
import pickle as pickle
from pathlib import Path

# temp
import time
import matplotlib.pyplot as plt

# TODO this is currently broken, but can be fixed once ilastik python version is bumped
# pysh.backends.select_preferred_backend(backend='ducc', nthreads=1)

logger = logging.getLogger(__name__)

_condition = threading.RLock()


class SphericalProjection(ObjectFeaturesPlugin):
    ndim = None
    margin = 0  # necessary for calling compute_local

    # projection order is
    projectionorder = ["MAX projection", "MIN projection", "SHAPE projection", "MEAN projection"]
    scaleorder = [
        "low degrees",
        "high degrees (undersampled)",
        "high degrees",
    ]  # NOTE this is redefined in fill_properties

    projections = np.zeros(len(projectionorder), dtype=bool)  # contains selection of which projections should be done
    features = None

    # Set in compute_local on first run:
    raysLUT = None
    scale = 80
    fineness = int(np.pi * scale)

    def availableFeatures(self, image, labels):

        if labels.ndim != 3:
            return {}

        names = []
        result = {}
        for proj in self.projectionorder:
            for ix, scale in enumerate(self.scaleorder):
                name = proj + " - " + scale
                result[name] = {}

        result = self.fill_properties(result)
        for f, v in result.items():
            v["tooltip"] = f
        return result

    @staticmethod
    def fill_properties(features):
        # fill in the detailed information about the features.
        # features should be a dict with the feature_name as key.
        # NOTE, this function needs to be updated every time skeleton features change
        scaleorder = ["low degrees", "high degrees (undersampled)", "high degrees"]
        for name, feature in features.items():
            proj, scl = [part.strip() for part in name.split("-")]
            feature["group"] = proj
            feature["margin"] = 0
            if scl == scaleorder[0]:
                feature["displaytext"] = "coarse details"
                feature[
                    "detailtext"
                ] = f"Spherical harmonic degrees up to n of a {proj} spherical projection of the data"
            if scl == scaleorder[1]:
                feature["displaytext"] = "fine details (averaged)"
                feature[
                    "detailtext"
                ] = f"Log2 undersampled spherical harmonic degrees from n of a {proj} spherical projection of the data"
            if scl == scaleorder[2]:
                feature["displaytext"] = "fine details"
                feature[
                    "detailtext"
                ] = f"Spherical harmonic degrees from n of a {proj} spherical projection of the data; Adds many features."
        return features

    def unwrap_and_expand(self, image, binary_bbox, axes, features):

        t0 = time.time()
        rawbbox = image
        mask_object = binary_bbox

        # TODO reset conditionality here
        self.raysLUT = self.get_ray_table()

        cube = resize(
            image, (self.scale, self.scale, self.scale), preserve_range=True, order=1
        )  # this normalizes the data
        mask_cube = resize(img_as_bool(mask_object), (self.scale, self.scale, self.scale), order=0)
        segmented_cube = np.where(mask_cube, cube, -1)
        t1 = time.time()
        # if np.count_nonzero(mask_object) > 30:
        unwrapped = lookup_spherical(segmented_cube, self.raysLUT, self.fineness, self.projections)
        # else:  # artefacts of cellpose and other segmentation things are filtered out for time
        #     unwrapped = np.zeros((self.fineness + 1, self.fineness * 2 + 1, np.sum(self.projections)), dtype=np.float64)
        t2 = time.time()

        result = {}
        projectedix = 0
        for which_proj, projected in enumerate(self.projections):
            if projected:
                projection = unwrapped[:, :, projectedix].astype(float)
                # print(self.projectionorder[which_proj], np.max(projection), np.min(projection))
                # tifffile.imwrite("/Users/oanegros/Documents/screenshots/tmp_unwrapped4/"
                #     + str(t0)
                #     + "_"
                #     + str(np.count_nonzero(mask_object))
                #     + "_"
                #     + str(np.count_nonzero(mask_object == 0))
                #     + self.projectionorder[which_proj]
                #     + "unwrapGLQ_masked.tif",
                #     (projection*100).astype(np.int16), imagej=True)
                plt.imsave(
                    "/Users/oanegros/Documents/screenshots/tmp_unwrapped4/"
                    + str(t0)
                    + "_"
                    + str(np.count_nonzero(mask_object))
                    + "_"
                    + str(np.count_nonzero(mask_object == 0))
                    + self.projectionorder[which_proj]
                    + "unwrapGLQ_masked.png",
                    projection,
                )
                projectedix += 1

                zero, w = pysh.expand.SHGLQ(self.fineness)
                with _condition:  # shtools backend is not thread-safec
                    coeffs = pysh.expand.SHExpandGLQ(projection, w=w, zero=zero)

                power_per_dlogl = spectrum(coeffs, unit="per_dlogl", base=2)

                # bin in 2log spaced bins
                bins = np.logspace(0, np.log2(len(power_per_dlogl)), num=20, base=2, endpoint=True)
                bin_ix, current_bin, means = 0, [], []
                for degree, power in enumerate(power_per_dlogl[1:]):
                    current_bin.append(power)
                    if degree + 1 >= bins[bin_ix]:
                        if len(current_bin) > 0:  # this is for high ratio bins/degrees
                            means.append(np.mean(current_bin))
                        bin_ix += 1
                        current_bin = []
                if self.projectionorder[which_proj] + " - " + self.scaleorder[0] in self.features:
                    result[self.projectionorder[which_proj] + " - " + self.scaleorder[0]] = power_per_dlogl[1:9]
                if self.projectionorder[which_proj] + " - " + self.scaleorder[1] in self.features:
                    result[self.projectionorder[which_proj] + " - " + self.scaleorder[1]] = means[8:]
                if self.projectionorder[which_proj] + " - " + self.scaleorder[2] in self.features:
                    result[self.projectionorder[which_proj] + " - " + self.scaleorder[1]] = power_per_dlogl[9:]

        t3 = time.time()
        print("time to do full unwrap and expand: \t", t3 - t0)
        return result

    def _do_3d(self, image, binary_bbox, features, axes):
        results = []
        features = list(features.keys())
        results.append(self.unwrap_and_expand(image, binary_bbox, axes, features))
        return results[0]

    def compute_local(self, image, binary_bbox, features, axes):
        for feature in features:
            for ix, proj in enumerate(self.projectionorder):
                if proj == features[feature]["group"]:
                    self.projections[ix] = True
        self.features = features
        orig_bbox = binary_bbox

        np.nonzero(orig_bbox)
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
        # loading requires retyping of the dictionary for numba, which slows it down (see: https://github.com/numba/numba/issues/8797)
        try:
            t0 = time.time()
            name = "sphericalLUT" + str(self.scale) + ".pickle"
            with open(Path(__file__).parent / name, "rb") as handle:
                newLUT = pickle.load(handle)
            typed_rays = typed.Dict.empty(
                key_type=typeof((1, 1)),
                value_type=typeof(np.zeros((1, 3), dtype=np.int32)),
            )
            for k, v in newLUT.items():
                typed_rays[k] = v
            print("loaded ray table in: ", time.time() - t0)
            return typed_rays
        except Exception as e:
            print("recalculating LUT")
            t0 = time.time()
            rays = typed.Dict.empty(
                key_type=typeof((1, 1)),
                value_type=typeof(np.zeros((1, 3), dtype=np.int32)),
            )
            fill_ray_table(self.scale, GLQGridCoord(self.fineness), rays)
            self.save_ray_table(rays)
            t1 = time.time()
            print("time to make ray table: ", t1 - t0)
            return rays


# All numba-accelerated functions cannot receive self, so are not class functions


@jit(nopython=True)
def lookup_spherical(img, raysLUT, fineness, projections):
    print("lookup")
    unwrapped = np.zeros((fineness + 1, fineness * 2 + 1, np.sum(projections)), dtype=np.float64)
    for loc, ray in raysLUT.items():
        # TODO update indexing to "values = img[ray[:,0],ray[:,1],ray[:,2]]" or similar once numba multiple advanced indexing is merged https://github.com/numba/numba/pull/8491
        values = np.zeros(ray.shape[0])
        for ix, voxel in enumerate(ray):
            values[ix] = img[voxel[0], voxel[1], voxel[2]]
            if values[ix] < 0:  # quit when outside of object mask -  all outside of mask are set to -1
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
            vec = ray[0].astype(np.float64) - ray[len(values) - 1].astype(np.float64)
            vec -= vec < 0  # integer flooring issues
            unwrapped[loc[1], loc[0], proj] = np.linalg.norm(vec)
            proj += 1
        if projections[3]:  # MEAN
            unwrapped[loc[1], loc[0], proj] = np.sum(values) / len(values)
            # unwrapped[loc[1], loc[0], proj] = np.sum(values) / (unwrapped[loc[1], loc[0], proj - 1])
            proj += 1
    print("done with lookup")
    return unwrapped


# ---- only used in generating LUT ----


@jit(nopython=True)
def fill_ray_table(scale, GLQcoords, rays):
    centroid = np.array([scale, scale, scale], dtype=np.float32) / 2.0
    # centroid += 0.01 # slightly offset to reduce float rounding errors at the border
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
            pixels = march(ray, centroid, scale, marchlen=0.003).astype(np.int32).copy()

            # find unique pixels and keep order

            # NUMBA-fied version of: different = np.any(pixels[:-1, :] - pixels[1:, :],axis=1)
            different = np.array([np.any(difference) for difference in (pixels[:-1, :] - pixels[1:, :])])

            nz = np.nonzero(different)[0]
            unique_pixels = np.zeros((nz.shape[0] + 1, 3), dtype=np.int32)
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
    bounds = [np.array([scale, scale, scale]).astype(np.float64) - 0.4, np.array([0.0, 0.0, 0.0])]
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
    return np.inf  # parallel ray and plane
