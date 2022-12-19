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

# Written by Oane Gros WIP
###############################################################################
from ilastik.plugins import ObjectFeaturesPlugin
from ilastik.plugins_default.convex_hull_feature_description import fill_feature_description
import ilastik.applets.objectExtraction.opObjectExtraction

import vigra
import numpy
import logging

import numpy as np
from numba import jit, typeof, typed, types
from skimage.transform import resize
from pyshtools.expand import SHExpandDH
from pyshtools.spectralanalysis import spectrum
from numba.extending import overload, register_jitable
from numba.core.errors import TypingError

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


class TextureSphericalMIP250(ObjectFeaturesPlugin):
    local_preffix = "Spherical MIP harmonics 250 "  # note the space at the end, it's important #TODO why????
    fineness = 250
    ndim = None
    margin = 0
    raysLUT = None
    scale = int(fineness / np.pi)

    def availableFeatures(self, image, labels):

        if labels.ndim == 3:
            names = ["wave_" + str(i + 1).zfill(3) for i in range(self.fineness)]  # TODO check if this should be a list

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
        print("Im inside textureshperickalmip fill properties!!!!! ")
        # fill in the detailed information about the features.
        # features should be a dict with the feature_name as key.
        # NOTE, this function needs to be updated every time skeleton features change
        for feature in features:
            features[feature]["displaytext"] = feature
            features[feature][
                "detailtext"
            ] = " wave number of the spherical harmonics decomposition if a spherical maximum intensity projection from center is taken as a unit sphere circle."
            features[feature]["margin"] = 0  # needs to be set to trigger compute_local
        return features

    def unwrap_and_expand(self, image, label_bboxes, axes):
        rawbbox = image
        mask_object, mask_both, mask_neigh = label_bboxes

        if self.raysLUT == None:
            print("recalculating LUT")  # TODO move to generate ray table functions
            rays = typed.Dict.empty(
                key_type=typeof((0.0, 0.0)),
                value_type=typeof(np.zeros((1, 3), dtype=np.int16)),  # base the d2 instance values of the type of d1
            )
            t0 = time.time()
            self.raysLUT = generate_ray_table(self.fineness, self.scale, rays)
            t1 = time.time()
            print("time to make ray tayble: ", t1 - t0)

        segmented = np.where(np.invert(mask_object), image, 0)
        segmented_cube = resize(segmented, (self.scale, self.scale, self.scale), preserve_range=True)

        t0 = time.time()
        unwrapped = lookup_spherical(segmented_cube, self.raysLUT, self.fineness).T
        t1 = time.time()
        print("time to lookup ray tayble: ", t1 - t0)

        coeffs = SHExpandDH(unwrapped, sampling=2)
        power_per_dlogl = spectrum(coeffs, unit="per_dlogl")
        t2 = time.time()
        print("time to do spherical harmonics: ", t2 - t1)
        wavenames = ["wave_" + str(i + 1).zfill(3) for i in range(self.fineness)]
        result = {}
        for ix, wavename in enumerate(wavenames):
            result[wavename] = power_per_dlogl[ix]
        return result

    def _do_3d(self, image, label_bboxes, features, axes):
        print("in do3d")
        kwargs = locals()
        del kwargs["self"]
        del kwargs["features"]
        kwargs["label_bboxes"] = kwargs.pop("label_bboxes")
        results = []
        features = list(features.keys())
        results.append(self.unwrap_and_expand(image, label_bboxes, axes))
        return results[0]

    def compute_local(self, image, binary_bbox, features, axes):
        print("in compute local of spherical mip")
        margin = ilastik.applets.objectExtraction.opObjectExtraction.max_margin({"": features})
        passed, excl = ilastik.applets.objectExtraction.opObjectExtraction.make_bboxes(binary_bbox, margin)
        return self.do_channels(
            self._do_3d, image, label_bboxes=[binary_bbox, passed, excl], features=features, axes=axes
        )


@jit(
    nopython=True
)  # can not easily jit this as np.unique(axis=0) is not supported so needs some aux helper functions at the end of the doc
def generate_ray_table(fineness, scale, rays):
    fineness = fineness * 4
    dummy = np.zeros((scale, scale, scale), dtype=np.int16)
    centroid = np.array(dummy.shape, dtype=np.float32) / 2.0
    pi2range = np.linspace(-0.5 * np.pi, 1.5 * np.pi, fineness)
    pirange = np.linspace(-1 * np.pi, 0 * np.pi, int(fineness / 2))

    for phi_ix, phi in enumerate(pi2range):
        for theta_ix, theta in enumerate(pirange):
            ray = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], dtype=np.float64)
            pixels = nb_unique(march(ray, centroid, dummy, marchlen=0.4), axis=0)[0]
            rays[(phi, theta)] = pixels
    return rays


@jit(nopython=True)
def lookup_spherical(data_rescaled, raysLUT, fineness):
    fineness = fineness * 4
    unwrapped = np.zeros((fineness, int(fineness / 2)), dtype=np.float64)
    pi2range = np.linspace(-0.5 * np.pi, 1.5 * np.pi, fineness)
    pirange = np.linspace(-1 * np.pi, 0 * np.pi, int(fineness / 2))
    for phi_ix, phi in enumerate(pi2range):
        for theta_ix, theta in enumerate(pirange):
            # voxels = raysLUT[phi][theta]
            values = np.zeros(raysLUT[(phi, theta)].shape[0])
            for ix, voxel in enumerate(raysLUT[(phi, theta)]):
                values[ix] = data_rescaled[voxel[0], voxel[1], voxel[2]]
            unwrapped[phi_ix, theta_ix] = np.amax(values)
    return unwrapped


@jit(nopython=True)
def project_spherical(ch_data, centroid, fineness):
    fineness = fineness * 4  # TODO refer to self.fineness for all instances - define by max wave_n
    unwrapped = np.zeros((fineness, int(fineness / 2)), dtype=np.float64)
    pi2range = np.linspace(-0.5 * np.pi, 1.5 * np.pi, fineness)
    pirange = np.linspace(-1 * np.pi, 0 * np.pi, int(fineness / 2))
    for phi_ix, phi in enumerate(pi2range):
        for theta_ix, theta in enumerate(pirange):
            unwrapped[phi_ix, theta_ix] = project_ray(ch_data, centroid, theta, phi)
    return unwrapped


@jit(nopython=True)
def project_ray(ch_data, centroid, theta, phi, ellipse=True):
    ray = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], dtype=np.float64)
    # ray *= [meta_info['PhysicalSizeX'], meta_info['PhysicalSizeY'], meta_info['PhysicalSizeZ']]
    if ellipse:
        ray *= np.array(ch_data.shape)
    ray /= np.linalg.norm(ray)
    pixels = march(ray, centroid, ch_data)
    intensities = np.zeros(int(est_length))
    for ix, pixel in enumerate(pixels):
        intensities[ix] = data[pixel[0], pixel[1], pixel[2]]
    # return phi
    return np.amax(intensities)


@jit(nopython=True)
def march(ray, centroid, data, marchlen=0.6):
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
    pixels = np.stack(pixels).astype(np.int16).T
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


@jit(nopython=True, cache=True)
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
