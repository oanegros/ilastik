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

# Plugin written by Oane Gros WIP
###############################################################################

# core ilastik
from ilastik.plugins import ObjectFeaturesPlugin
from ilastik.plugins_default.convex_hull_feature_description import fill_feature_description
import ilastik.applets.objectExtraction.opObjectExtraction

# core
import vigra
import numpy
import logging
import numpy as np

# transformations and speedup
from numba import jit, typeof, typed, types
from skimage.transform import resize
from skimage import img_as_bool
from pyshtools.expand import SHExpandDH
from pyshtools.spectralanalysis import spectrum
from numba.extending import overload, register_jitable
from numba.core.errors import TypingError

# saving/loading LUT
import pickle as pickle
from pathlib import Path

# temp
import time

logger = logging.getLogger(__name__)


def cleanup_value(val, nObjects):
    """ensure that the value is a numpy array with the correct shape."""

    if type(val) == list:
        return val

    val = numpy.asarray(val)

    if val.ndim == 1:
        val = val.reshape(-1, 1)

    assert val.shape[0] == nObjects
    # remove background
    val = val[1:]
    return val


def cleanup(d, nObjects, features):
    result = dict((k, cleanup_value(v, nObjects)) for k, v in d.items())
    newkeys = set(result.keys()) & set(features)
    return dict((k, result[k]) for k in newkeys)


class SphericalProjection(ObjectFeaturesPlugin):
    local_preffix = "Spherical projections "  # note the space at the end, it's important #TODO why???? - this comment was in another file
    ndim = None
    margin = 0
    projectionorder = ["MAX projection", "MIN projection", "SUM projection", "MEAN projection"]  # nee

    # Set in compute_local on first run:
    raysLUT = None
    fineness = None
    scale = 0
    projections = np.zeros(4, dtype=bool)

    def availableFeatures(self, image, labels):

        if labels.ndim == 3:
            names = [  # compute_local actually uses the last number to set self.scale, so be careful changing
                "resolution 10x10x10",
                "resolution 20x20x20",
                "resolution 40x40x40",
                "resolution 80x80x80",
            ]
            for proj in self.projectionorder:
                names.append(proj)
            tooltips = {}
            result = dict((n, {}) for n in names)
            result = self.fill_properties(result)
            for f, v in result.items():
                v["tooltip"] = self.local_preffix + f
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

    def unwrap_and_expand(self, image, label_bboxes, axes, features):
        t0 = time.time()
        rawbbox = image
        mask_object, mask_both, mask_neigh = label_bboxes

        if self.raysLUT == None:
            self.raysLUT = self.get_ray_table()

        cube = resize(image, (self.scale, self.scale, self.scale), preserve_range=True)
        mask_cube = img_as_bool(resize(mask_object, (self.scale, self.scale, self.scale), order=0))
        segmented_cube = np.where(np.invert(mask_cube), cube, -1)
        # necessary to declare typed dictionary for Numba

        t1 = time.time()
        unwrapped = lookup_spherical(segmented_cube, self.raysLUT, self.fineness, self.projections)

        t2 = time.time()
        print("time to lookup ray tayble: ", t2 - t1)
        result = {}
        projectedix = 0
        for which_proj, projected in enumerate(self.projections):
            if projected:
                projection = unwrapped[:, :, projectedix]
                projectedix += 1  #
                coeffs = SHExpandDH(projection, sampling=2)
                power_per_dlogl = spectrum(coeffs, unit="per_dlogl")
                result[self.projectionorder[which_proj]] = power_per_dlogl[1:]

        t3 = time.time()
        print("time to do full unwrap and expand: ", t3 - t0)
        # print(result)
        return result

    def _do_3d(self, image, label_bboxes, features, axes):
        print("in do3d")
        results = []
        features = list(features.keys())
        results.append(self.unwrap_and_expand(image, label_bboxes, axes, features))
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
        margin = ilastik.applets.objectExtraction.opObjectExtraction.max_margin({"": features})
        passed, excl = ilastik.applets.objectExtraction.opObjectExtraction.make_bboxes(binary_bbox, margin)
        return self.do_channels(
            self._do_3d, image, label_bboxes=[binary_bbox, passed, excl], features=features, axes=axes
        )

    def save_ray_table(self, rays):
        outLUT = {}  # need to un-type the dictionary for pickling
        for k, v in rays.items():
            outLUT[k] = v
        outLUT["metadata"] = {"fineness": self.fineness}  # add more if distinguishing settings are made
        name = "sphericalLUT" + str(self.scale) + ".pickle"
        with open(Path(__file__).parent / name, "wb") as ofs:
            pickle.dump(outLUT, ofs, protocol=pickle.HIGHEST_PROTOCOL)
        return

    def get_ray_table(self):
        try:
            t0 = time.time()
            name = "sphericalLUT" + str(self.scale) + ".pickle"
            with open(Path(__file__).parent / name, "rb") as handle:
                newLUT = pickle.load(handle)
            metadata = newLUT.pop("metadata")
            assert metadata["fineness"] == self.fineness
            typed_rays = typed.Dict.empty(
                key_type=typeof((1, 1)),
                value_type=typeof(np.zeros((1, 3), dtype=np.int16)),
            )
            for k, v in newLUT.items():
                typed_rays[k] = v
            print("loaded ray table of fineness ", metadata["fineness"], " in: ", time.time() - t0)
            return typed_rays
        except:
            return self.generate_ray_table()

    def generate_ray_table(self):
        # do all tasks outside of numba handling
        print("recalculating LUT")
        t0 = time.time()
        prerays = typed.Dict.empty(
            key_type=typeof((1, 1)),
            value_type=typeof(np.zeros((1, 3), dtype=np.float64)),
        )
        fill_ray_table(self.fineness, self.scale, prerays)
        # rounding instead of flooring is hard within numba, apparently.
        rays = typed.Dict.empty(
            key_type=typeof((1, 1)),
            value_type=typeof(np.zeros((1, 3), dtype=np.int16)),
        )
        for coord, ray in prerays.items():
            rays[coord] = np.round(ray).astype(np.int16)
        self.save_ray_table(rays)
        t1 = time.time()
        print("time to make ray table: ", t1 - t0)
        return rays


@jit(nopython=True)
def fill_ray_table(fineness, scale, rays):
    # needs helper functions for np.unique and np.all to jit :(
    dummy = np.zeros((scale, scale, scale), dtype=np.int16)
    centroid = np.array(dummy.shape, dtype=np.float32) / 2.0
    pi2range = np.linspace(-0.5 * np.pi, 1.5 * np.pi, fineness * 4)
    pirange = np.linspace(-1 * np.pi, 0 * np.pi, fineness * 2)

    for phi_ix, phi in enumerate(pi2range):
        for theta_ix, theta in enumerate(pirange):
            ray = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], dtype=np.float64)
            pixels = nb_unique(march(ray, centroid, dummy, marchlen=0.3), axis=0)[0]
            rays[(phi_ix, theta_ix)] = pixels
    return rays


@jit(nopython=True)
def lookup_spherical(data_rescaled, raysLUT, fineness, projections):
    unwrapped = np.zeros((fineness * 2, fineness * 4, np.sum(projections)), dtype=np.float64)
    for k, v in raysLUT.items():
        values = np.zeros(v.shape[0])
        for ix, voxel in enumerate(v):
            values[ix] = data_rescaled[voxel[0], voxel[1], voxel[2]]
            if values[ix] < 0:
                # quit when outside of object mask - assumes convex shape
                break
        proj = 0
        if projections[0]:  # MAX
            unwrapped[k[1], k[0], proj] = np.amax(values)
            proj += 1
        if projections[1]:  # MIN
            unwrapped[k[1], k[0], proj] = np.amin(values)
            proj += 1
        if projections[2]:  # MEAN
            unwrapped[k[1], k[0], proj] = np.mean(values)
            proj += 1
        if projections[3]:  # SUM
            unwrapped[k[1], k[0], proj] = np.sum(values)
            proj += 1
    return unwrapped


@jit(nopython=True)
def march(ray, centroid, data, marchlen):
    increment = ray * marchlen
    distances = []
    normals = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
    bounds = [np.array(data.shape) - np.array([1.0, 1.0, 1.0]), np.array([0.0, 0.0, 0.0])]
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
    # pixels = np.around(np.stack(pixels)).T
    pixels = np.stack(pixels).T
    # pixels = np.round_(pixels)
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


# ----  numba helper functions ----
# taken from https://github.com/numba/numba/issues/7663, by https://github.com/rishi-kulkarni


@jit(nopython=True)
def nb_unique(input_data, axis=0):
    """2D np.unique(a, return_index=True, return_counts=True)

    Parameters
    ----------
    input_data : 2D numeric array
    axis : int, optional
        axis along which to identify unique slices, by default 0
    Returns
    -------
    2D array
        unique rows (or columns) from the input array
    1D array of ints
        indices of unique rows (or columns) in input array
    1D array of ints
        number of instances of each unique row
    """

    # don't want to sort original data
    if axis == 1:
        data = input_data.T.copy()

    else:
        data = input_data.copy()

    # so we can remember the original indexes of each row
    orig_idx = np.array([i for i in range(data.shape[0])])

    # sort our data AND the original indexes
    for i in range(data.shape[1] - 1, -1, -1):
        sorter = data[:, i].argsort(kind="mergesort")

        # mergesort to keep associations
        data = data[sorter]
        orig_idx = orig_idx[sorter]
    # get original indexes
    idx = [0]

    if data.shape[1] > 1:
        bool_idx = ~np.all((data[:-1] == data[1:]), axis=1)
        additional_uniques = np.nonzero(bool_idx)[0] + 1

    else:
        additional_uniques = np.nonzero(~(data[:-1] == data[1:]))[0] + 1

    idx = np.append(idx, additional_uniques)
    # get counts for each unique row
    counts = np.append(idx[1:], data.shape[0])
    counts = counts - idx
    return data[idx], orig_idx[idx], counts


@overload(np.all)
def np_all(x, axis=None):

    # ndarray.all with axis arguments for 2D arrays.

    @register_jitable
    def _np_all_axis0(arr):
        out = np.logical_and(arr[0], arr[1])
        for v in iter(arr[2:]):
            for idx, v_2 in enumerate(v):
                out[idx] = np.logical_and(v_2, out[idx])
        return out

    @register_jitable
    def _np_all_flat(x):
        out = x.all()
        return out

    @register_jitable
    def _np_all_axis1(arr):
        out = np.logical_and(arr[:, 0], arr[:, 1])
        for idx, v in enumerate(arr[:, 2:]):
            for v_2 in iter(v):
                out[idx] = np.logical_and(v_2, out[idx])
        return out

    if isinstance(axis, types.Optional):
        axis = axis.type

    if not isinstance(axis, (types.Integer, types.NoneType)):
        raise TypingError("'axis' must be 0, 1, or None")

    if not isinstance(x, types.Array):
        raise TypingError("Only accepts NumPy ndarray")

    if not (1 <= x.ndim <= 2):
        raise TypingError("Only supports 1D or 2D NumPy ndarrays")

    if isinstance(axis, types.NoneType):

        def _np_all_impl(x, axis=None):
            return _np_all_flat(x)

        return _np_all_impl

    elif x.ndim == 1:

        def _np_all_impl(x, axis=None):
            return _np_all_flat(x)

        return _np_all_impl

    elif x.ndim == 2:

        def _np_all_impl(x, axis=None):
            if axis == 0:
                return _np_all_axis0(x)
            else:
                return _np_all_axis1(x)

        return _np_all_impl

    else:

        def _np_all_impl(x, axis=None):
            return _np_all_flat(x)

        return _np_all_impl
