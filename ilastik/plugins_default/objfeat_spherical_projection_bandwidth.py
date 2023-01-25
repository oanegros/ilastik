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


# def cleanup_value(val, nObjects):
#     """ensure that the value is a numpy array with the correct shape."""

#     if type(val) == list:
#         return val

#     val = numpy.asarray(val)

#     if val.ndim == 1:
#         val = val.reshape(-1, 1)

#     assert val.shape[0] == nObjects
#     # remove background
#     val = val[1:]
#     return val


# def cleanup(d, nObjects, features):
#     result = dict((k, cleanup_value(v, nObjects)) for k, v in d.items())
#     newkeys = set(result.keys()) & set(features)
#     return dict((k, result[k]) for k in newkeys)


class SphericalProjection(ObjectFeaturesPlugin):
    local_preffix = "Spherical projections "
    ndim = None
    margin = 0

    projectionorder = ["MAX", "MIN", "SUM", "MEAN"]  # this is used to index the projection array
    resolutions = [0, 10, 20, 40, 80]  # 0 start to make cumulative methods work easier

    # Set in compute_local on first run:
    raysLUTdct = {}
    featdct = {}

    def availableFeatures(self, image, labels):
        if labels.ndim == 3:
            # projectionorder = self.projectionorder
            # resolutions = self.resolutions
            names = []
            for proj in self.projectionorder:
                for ix, res in enumerate(self.resolutions[1:]):
                    prevres = self.resolutions[ix]
                    names.append(f"{proj}_{res}")
            result = dict((n, {}) for n in names)
            result = self.fill_properties(result)
            for f, v in result.items():
                v["tooltip"] = self.local_preffix + f
        else:
            result = {}

        return result

    def fill_properties(self, names):
        # fill in the detailed zinformation about the features.
        # features should be a dict with the feature_name as key.
        # NOTE, this function needs to be updated every time skeleton features change
        features = {}
        for proj in self.projectionorder:
            for ix, res in enumerate(self.resolutions[1:]):
                prevres = self.resolutions[ix]
                feature = f"{proj}_{res}"
                features[feature] = {}
                features[feature]["displaytext"] = f"{proj} texture {prevres}x{prevres}x{prevres} to {res}x{res}x{res}"
                features[feature][
                    "detailtext"
                ] = f"Texture information between {prevres}x{prevres}x{prevres} and {res}x{res}x{res} 3D resolution of spherical {proj} intensity projection.\n"
                features[feature]["group"] = proj + " projection"
                features[feature]["margin"] = 0  # needs to be set to trigger compute_local
        return features

    def unwrap_and_expand(self, image, label_bboxes, axes, projections, scale):
        cube = resize(image, (scale, scale, scale), preserve_range=True)
        mask_cube = img_as_bool(resize(label_bboxes[0], (scale, scale, scale), order=0))
        segmented_cube = np.where(mask_cube, cube, -1)

        unwrapped = lookup_spherical(segmented_cube, self.raysLUTdct[scale], projections)

        result = {}
        projectedix = 0

        for which_proj, projected in enumerate(projections):
            if projected:
                projection = unwrapped[:, :, projectedix]
                projectedix += 1
                coeffs = SHExpandDH(projection, sampling=2)
                power_per_dlogl = spectrum(coeffs, unit="per_dlogl")
                # starts from previous last degree (pi*self.resolutions[scale_ix-1]) onward
                result[f"{self.projectionorder[which_proj]}_{scale}"] = power_per_dlogl[
                    int(self.resolutions[self.resolutions.index(scale) - 1] * np.pi) :
                ]
        return result

    def _do_3d(self, image, label_bboxes, featdct, axes):
        t0 = time.time()
        results = {}
        for scale, projections in featdct.items():
            results.update(self.unwrap_and_expand(image, label_bboxes, axes, projections, scale))
        t3 = time.time()
        print("time to do full bandwidthed unwrap and expand: ", t3 - t0)
        return results

    def compute_local(self, image, binary_bbox, features, axes):
        margin = ilastik.applets.objectExtraction.opObjectExtraction.max_margin({"": features})
        margin = ilastik.applets.objectExtraction.opObjectExtraction.max_margin({"": features})
        margin_max = [image.shape[i] - margin[i] for i in range(len(margin))]
        image = image[margin[0] : margin_max[0] - 1, margin[1] : margin_max[1] - 1, margin[2] : margin_max[2] - 1]
        binary_bbox = binary_bbox[
            margin[0] : margin_max[0] - 1, margin[1] : margin_max[1] - 1, margin[2] : margin_max[2] - 1
        ]
        passed, excl = ilastik.applets.objectExtraction.opObjectExtraction.make_bboxes(binary_bbox, margin)

        # reorder features to np array by resolution to allow for efficient calculation
        featdct = {}
        for feature in features:
            proj, res = feature.split("_")
            res = int(res)
            if res not in featdct:
                featdct[res] = np.zeros(len(self.projectionorder), dtype=bool)
                if res not in self.raysLUTdct:
                    self.raysLUTdct[res] = self.get_ray_table(res)
            featdct[res][self.projectionorder.index(proj)] = True

        return self.do_channels(
            self._do_3d, image, label_bboxes=[binary_bbox, passed, excl], featdct=featdct, axes=axes
        )

    def save_ray_table(self, rays, scale):
        # save a pickle of the rayLUT, requires retyping of the dictionary
        # this is because the rays are ragged, and not single-length
        # TODO an attempt could be made for 3d-np array of rays with -1 values after ending ray
        outLUT = {}  # need to un-type the dictionary for pickling
        for k, v in rays.items():
            outLUT[k] = v
        name = "sphericalLUT" + str(scale) + ".pickle"
        with open(Path(__file__).parent / name, "wb") as ofs:
            pickle.dump(outLUT, ofs, protocol=pickle.HIGHEST_PROTOCOL)
        return

    def get_ray_table(self, scale):
        # try to load or generate new
        # loading requires retyping of the dictionary for numba, which slows it down
        try:
            t0 = time.time()
            name = "sphericalLUT" + str(scale) + ".pickle"
            with open(Path(__file__).parent / name, "rb") as handle:
                newLUT = pickle.load(handle)
            typed_rays = typed.Dict.empty(
                key_type=typeof((1, 1)),
                value_type=typeof(np.zeros((1, 3), dtype=np.int16)),
            )
            for k, v in newLUT.items():
                typed_rays[k] = v
            print("loaded ray table of scale ", scale, " in: ", time.time() - t0)
            return typed_rays
        except:
            return self.generate_ray_table(scale)

    def generate_ray_table(self, scale):
        # make new ray table = do all tasks outside of numba handling
        print("recalculating LUT")
        t0 = time.time()
        prerays = typed.Dict.empty(
            key_type=typeof((1, 1)),
            value_type=typeof(np.zeros((1, 3), dtype=np.float64)),
        )
        print(scale)
        fill_ray_table(scale, prerays)
        # rounding instead of flooring is hard within numba, apparently.
        rays = typed.Dict.empty(
            key_type=typeof((1, 1)),
            value_type=typeof(np.zeros((1, 3), dtype=np.int16)),
        )
        for coord, ray in prerays.items():
            rays[coord] = np.round(ray).astype(np.int16)
        self.save_ray_table(rays, scale)
        t1 = time.time()
        print("time to make ray table: ", t1 - t0)
        return rays


@jit(nopython=True)
def lookup_spherical(data_rescaled, raysLUT, projections):
    fineness = int(np.pi * data_rescaled.shape[0])
    unwrapped = np.zeros((fineness * 2, fineness * 4, np.sum(projections)), dtype=np.float64)
    for k, v in raysLUT.items():
        values = np.zeros(v.shape[0])
        for ix, voxel in enumerate(v):
            if values[ix] < 0:
                # quit when outside of object mask - assumes convex shape - all outside of mask are set to -1
                break
            values[ix] = data_rescaled[voxel[0], voxel[1], voxel[2]]
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
    return np.log2(unwrapped)


# ---- only used in generating LUT ----


@jit(nopython=True)
def fill_ray_table(scale, rays):
    # needs helper functions for np.unique and np.all to jit
    fineness = int(np.pi * scale)
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


# f"The largest selected resolution defines computation speed. \n\n"+\
# f"In spherical projections this is the order of operations: \n"+\
# f"1. Each object is rescaled to the Max selected resolution \n"+\
# f"2. Rays are cast from the centroid, saving the {proj} intensity \n"+\
# f"3. Spherical projections are decomposed into spherical harmonics (analogous to Fourier expansion, for the sphere)\n"+\
# f"4. Rotationally invariant power spectrum (per 2_log(degree)) are saved. \n"+\
# f"5. This feature has degrees (analogous to wavelength) {int(prevres*np.pi)} to {int(res*np.pi)}."
