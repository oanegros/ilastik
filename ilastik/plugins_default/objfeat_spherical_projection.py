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

# fourier transformations and speedup
from numba import jit, typeof, typed, types
from skimage.transform import resize
from skimage.filters import gaussian
from skimage import img_as_bool
import pyshtools as pysh
from pyshtools.backends.shtools import GLQGridCoord
from pyshtools.expand import SHExpandGLQ
from pyshtools.spectralanalysis import spectrum
import scipy
import ducc0

import threading
import tifffile

# saving/loading LUT
import pickle as pickle
from pathlib import Path

# temp
import time
import matplotlib.pyplot as plt
import matplotlib as mpl

logger = logging.getLogger(__name__)

# _condition = threading.RLock()
pysh.backends.select_preferred_backend(backend="ducc", nthreads=5)


class SphericalProjection(ObjectFeaturesPlugin):
    ndim = None
    margin = 0  # necessary for calling compute_local

    # projection order is index in project_()
    projectionorder = [
        "MAX projection",
        "MIN projection",
        "SHAPE projection",
        "MEAN projection",
        "DIST_TO_MAX projection",
    ]
    detailorder = [
        "low degrees",
        "high degrees (undersampled)",
        "high degrees",
    ]  # NOTE this is redefined in fill_properties

    projections = np.zeros(len(projectionorder), dtype=bool)  # contains selection of which projections should be done
    features = None
    raysLUT = None
    bin_start, bin_ends, n_coarse = None, None, None
    ndim = 0

    # Hyperparameters
    scale = 80  # transforms to cube of size scale by scale by scale
    reduced_spectrum_length = 20  # length of coarse + fine details (averaged)

    def availableFeatures(self, image, labels):
        if labels.ndim < 2 or labels.ndim > 3:
            return {}
        self.ndim = labels.ndim

        names = []
        result = {}
        for proj in self.projectionorder:
            for ix, scale in enumerate(self.detailorder):
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
        detailorder = ["low degrees", "high degrees (undersampled)", "high degrees"]
        # detailorder needs to be the same as self.detailorder, but cannot inherit due to architectural issues
        for name, feature in features.items():
            proj, scl = [part.strip() for part in name.split("-")]
            feature["group"] = proj
            feature["margin"] = 0
            if scl == detailorder[0]:
                feature["displaytext"] = proj + " coarse details"
                feature[
                    "detailtext"
                ] = f"Spherical harmonic degrees up to n of a {proj} spherical projection of the data"
            if scl == detailorder[1]:
                feature["displaytext"] = proj + " fine details (averaged)"
                feature[
                    "detailtext"
                ] = f"Log2 undersampled spherical harmonic degrees from n of a {proj} spherical projection of the data"
            if scl == detailorder[2]:
                feature["displaytext"] = proj + " fine details (all)"
                feature[
                    "detailtext"
                ] = f"Spherical harmonic degrees from n of a {proj} spherical projection of the data; Adds many features."
        return features

    def unwrap_and_expand(self, image, binary_bbox, axes, features):
        t0 = time.time()
        rawbbox = image
        mask_object = binary_bbox

        # print(image.shape)
        # print(str(t0)+ "_"+ str(np.count_nonzero(mask_object)), image.shape)
        if self.raysLUT == None:
            self.raysLUT = self.get_ray_table(len(image.shape))
        # assert(len(next(iter(self.raysLUT))) == len(image.shape)-1) # switching dims is unsupported, but this to check

        # resizing of data is also done for 2D to make the code less convoluted
        cube = resize(image, (self.scale, self.scale, self.scale), preserve_range=False, order=1)
        # print(np.max(cube), np.min(cube))
        minval, maxval = np.min(cube), np.max(cube)
        if minval != maxval:
            cube -= minval
            cube *= 1 / maxval
        # print(np.max(cube), np.min(cube))
        cube = cube * 65536
        # print(np.max(cube), np.min(cube))
        mask_cube = resize(img_as_bool(mask_object), tuple([self.scale] * len(image.shape)), order=0)
        segmented_cube = np.where(mask_cube, cube, -1)
        t1 = time.time()

        # print(image.shape)
        unwrapped = lookup(segmented_cube, self.raysLUT, int(np.pi * self.scale), self.projections)

        t2 = time.time()

        # self.save_tifs(rawbbox, mask_object, segmented_cube, t0)

        result = {}
        projectedix = 0
        used_projections = [which_proj for which_proj, projected in enumerate(self.projections) if projected]
        for projectedix, projection in enumerate(unwrapped):
            which_proj = self.projectionorder[used_projections[projectedix]]

            projectedix += 1
            if self.ndim == 2:
                projection = projection[0, : int(self.scale * np.pi)]
                power = np.abs(scipy.fft.fft(projection))
            else:
                zero, w = pysh.expand.SHGLQ(int(np.pi * self.scale))
                # with _condition:  # shtools backend is not thread-safe, switch to ducc in future see issue #385 in shtools
                coeffs = pysh.expand.SHExpandGLQ(projection, w=w, zero=zero)
                power = spectrum(coeffs, unit="per_dlogl", base=2)[1:]

            # self.save_prjs(which_proj, spectrum, coeffs, projection, t0, mask_object)

            # bin higher degrees in 2log spaced bins:
            if self.n_coarse is None:
                self.get_bins(len(power))
            means = [np.mean(power[s:e]) for s, e in zip(self.bin_start, self.bin_ends)]
            # print(self.bin_start, self.bin_ends, self.n_coarse)
            # # Bin center values:
            # print(list(np.arange(0,self.n_coarse, dtype=float) ) + [np.mean([start,end]) for start, end in zip(self.bin_start, self.bin_ends)])

            if which_proj + " - " + self.detailorder[0] in self.features:
                result[which_proj + " - " + self.detailorder[0]] = power[: self.n_coarse]
            if which_proj + " - " + self.detailorder[1] in self.features:
                result[which_proj + " - " + self.detailorder[1]] = means
            if which_proj + " - " + self.detailorder[2] in self.features:
                result[which_proj + " - " + self.detailorder[2]] = power[self.n_coarse :]

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

        margin = [(np.min(dim), np.max(dim) + 1) for dim in np.nonzero(binary_bbox)]

        image = image[margin[0][0] : margin[0][1], margin[1][0] : margin[1][1], margin[2][0] : margin[2][1]]
        binary_bbox = binary_bbox[margin[0][0] : margin[0][1], margin[1][0] : margin[1][1], margin[2][0] : margin[2][1]]

        assert np.sum(orig_bbox) - np.sum(binary_bbox) == 0
        return self.do_channels(self._do_3d, image, binary_bbox=binary_bbox, features=features, axes=axes)

    def save_ray_table(self, fname, rays):
        # save a pickle of the rayLUT, requires retyping of the dictionary
        # this is because the rays are ragged, and not single-length
        outLUT = {}  # need to un-type the dictionary for pickling
        for k, v in rays.items():
            outLUT[k] = v
        with open(Path(__file__).parent / fname, "wb") as ofs:
            pickle.dump(outLUT, ofs, protocol=pickle.HIGHEST_PROTOCOL)
        return

    def get_bins(self, veclength):
        # increase bins until self.reduced_spectrum_length is hit with integer bins
        # all linearly scaled bins will be 'coarse features'
        n_bins = self.reduced_spectrum_length
        bins = np.unique(np.logspace(0, np.log2(veclength - 1), num=n_bins, base=2, endpoint=True).astype(int))
        while len(bins) < self.reduced_spectrum_length:
            n_bins += 1
            bins = np.unique(np.logspace(0, np.log2(veclength - 1), num=n_bins, base=2, endpoint=True).astype(int))

        self.n_coarse = np.argmax(bins - (np.arange(len(bins)) + 1) > 0)
        self.bin_ends = bins[self.n_coarse :]
        self.bin_start = np.roll(bins, 1)[self.n_coarse :]
        return

    def get_ray_table(self, ndim):
        # try to load or generate new
        # loading requires retyping of the dictionary for numba, which slows it down (see: https://github.com/numba/numba/issues/8797)
        fname = f"sphericalLUT{self.scale}_{ndim}D.pickle"
        try:
            t0 = time.time()
            with open(Path(__file__).parent / fname, "rb") as handle:
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
            fill_ray_table(self.scale, GLQGridCoord(int(np.pi * self.scale)), rays, ndim)
            self.save_ray_table(fname, rays)
            t1 = time.time()
            print("time to make ray table: ", t1 - t0)
            return rays

    def save_tifs(self, rawbbox, mask_object, segmented_cube, t0):
        # TIF SAVE
        saveable = segmented_cube
        saveable[saveable == -1] = 0
        saveable = (saveable * 255 / np.max(segmented_cube)).astype(np.uint8)
        tifffile.imwrite(
            "/Users/oanegros/Documents/screenshots/tmp/"
            + str(t0)
            + "_"
            + str(np.count_nonzero(mask_object))
            + "_"
            + str(np.count_nonzero(mask_object == 0))
            + "CELL_masked.tif",
            saveable,
            imagej=True,
            metadata={"axes": "zyx"},
        )

        saveable = rawbbox
        saveable = np.where(mask_object, rawbbox, 0)
        tifffile.imwrite(
            "/Users/oanegros/Documents/screenshots/tmp/"
            + str(t0)
            + "_"
            + str(np.count_nonzero(mask_object))
            + "_"
            + str(np.count_nonzero(mask_object == 0))
            + "CELL_small_masked.tif",
            saveable,
            imagej=True,
            metadata={"axes": "zyx"},
        )
        return

    def save_prjs(self, which_proj, spectrum, coeffs, projection, t0, mask_object):
        # PNG SAVE
        # print(projection)
        plt.imsave(
            "/Users/oanegros/Documents/screenshots/tmp/"
            + str(t0)
            + "_"
            + str(np.count_nonzero(mask_object))
            + "_"
            + str(np.count_nonzero(mask_object == 0))
            + which_proj
            + "unwrapGLQ_masked.png",
            resize(
                projection, (int(np.pi * self.scale) + 1, int(np.pi * self.scale) * 2 + 1), preserve_range=True, order=0
            ),
        )
        # # 1D Spectrum
        pysh.SHCoeffs.from_array(coeffs).to_file(
            "/Users/oanegros/Documents/screenshots/tmp/"
            + str(t0)
            + "_coeffs_"
            + str(np.count_nonzero(mask_object == 0))
            + which_proj
            + ".shtools",
        )

        # TIF SAVE PROJ
        projection = 65535 * ((projection - np.min(projection)) / (np.max(projection) - np.min(projection)))
        tifffile.imwrite(
            "/Users/oanegros/Documents/screenshots/tmp/"
            + str(t0)
            + "_"
            + str(np.count_nonzero(mask_object))
            + "_"
            + str(np.count_nonzero(mask_object == 0))
            + which_proj
            + "unwrapGLQ_masked.tif",
            projection.astype(np.uint16),
            imagej=True,
        )
        return


# All numba-accelerated functions cannot receive self, so are not class functions

# NOTE: lookup_spherical and lookup_circle are only split because fancy indexing is unsupported in numba
# and i wanted to avoid checking ndim fore each indexed voxel
@jit(nopython=True)
def lookup(img, raysLUT, fineness, projections):
    unwrapped = np.zeros((np.sum(projections), fineness + 1, fineness * 2 + 1), dtype=np.float64)
    for loc, ray in raysLUT.items():
        values = np.zeros(ray.shape[0])
        for ix, voxel in enumerate(ray):
            values[ix] = img[voxel[0], voxel[1], voxel[2]]
            if values[ix] < 0:  # quit when outside of object mask -  all outside of mask are set to -1
                if ix != 0:  # centroid is not outside of mask
                    values = values[:ix]
                    break
        proj = 0
        ray = ray.astype(np.float64)
        unwrapped[:, loc[1], loc[0]] = project_(ray, values, projections)
    return unwrapped


@jit(nopython=True)
def project_(ray, values, projections):
    vals = np.zeros(np.sum(projections), dtype=np.float64)
    proj = 0
    if projections[0]:  # MAX
        vals[proj] = np.amax(values)
        proj += 1
    if projections[1]:  # MIN
        vals[proj] = np.amin(values)
        proj += 1
    if projections[2]:  # SHAPE
        vec = ray[0].astype(np.float64) - ray[len(values) - 1].astype(np.float64)
        vec -= vec < 0  # integer flooring issues
        vals[proj] = np.linalg.norm(vec)
        proj += 1
    if projections[3]:  # MEAN
        vals[proj] = np.sum(values) / len(values)
        proj += 1
    if projections[4]:  # DIST_TO_MAX
        vec = ray[0].astype(np.float64) - ray[np.argmax(values)].astype(np.float64)
        vec -= vec < 0  # integer flooring issues
        vals[proj] = np.linalg.norm(vec)
        proj += 1
    return vals


# ---- only used in generating LUT ----


@jit(nopython=True)
def fill_ray_table(scale, GLQcoords, rays, ndim):
    centroid = np.array([scale, scale, scale], dtype=np.float32) / 2.0
    if ndim == 2:
        glq_lon = np.linspace(0, 2 * np.pi, int(scale * np.pi))
        glq_lat = np.array([0.0])
    else:
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


# print(self.projectionorder[which_proj], np.max(projection), np.min(projection))

# # TIF SAVE
# saveable = segmented_cube
# saveable[saveable == -1] = 0
# saveable = (saveable * 255 / np.max(segmented_cube) ).astype(np.uint8)
# tifffile.imwrite("/Users/oanegros/Documents/screenshots/tmp_2/"
#     + str(t0)
#     + "_"
#     + str(np.count_nonzero(mask_object))
#     + "_"
#     + str(np.count_nonzero(mask_object == 0))
#     + self.projectionorder[which_proj]
#     + "CELL_masked.tif",
#     saveable, imagej=True)

# # TIF SAVE PROJ
# tifffile.imwrite("/Users/oanegros/Documents/screenshots/tmp_2/"
#     + str(t0)
#     + "_"
#     + str(np.count_nonzero(mask_object))
#     + "_"
#     + str(np.count_nonzero(mask_object == 0))
#     + self.projectionorder[which_proj]
#     + "unwrapGLQ_masked.tif",
#     (projection * 255 / np.max(projection) ).astype(np.uint8), imagej=True)

# # PNG SAVE
# plt.imsave(
#     "/Users/oanegros/Documents/screenshots/tmp_2/"
#     + str(t0)
#     + "_"
#     + str(np.count_nonzero(mask_object))
#     + "_"
#     + str(np.count_nonzero(mask_object == 0))
#     + self.projectionorder[which_proj]
#     + "unwrapGLQ_masked.png",
#     projection,
# )

# # 1D Spectrum
# pysh.SHCoeffs.from_array(coeffs).plot_spectrum(
#     show=False,
#     unit="per_dlogl",
#     fname="/Users/oanegros/Documents/screenshots/tmp_2/"
#     + str(t0)
#     + "_spectrum_"
#     + str(np.count_nonzero(mask_object == 0))
#     + self.projectionorder[which_proj]
#     + ".svg",
# )
# # 2D spectrum
# pysh.SHCoeffs.from_array(coeffs).plot_spectrum2d(
#     show=False,
#     fname="/Users/oanegros/Documents/screenshots/tmp_2/"
#     + str(t0)
#     + "_spectrum2d_"
#     + str(np.count_nonzero(mask_object == 0))
#     + self.projectionorder[which_proj]
#     + ".png",
# )


# new_rc_params = {'text.usetex': False,
#     "svg.fonttype": 'none'
# }
# mpl.rcParams.update(new_rc_params)
# f, ax = plt.subplots(figsize=(7, 7))
# ax.set_xscale('log', base=2)
# ax.set_yscale('log', base=2)
# ax.plot(np.arange(1,251),power[1:])
# plt.savefig(
#     fname="/Users/oanegros/Documents/screenshots/tmp_2/"
#     + str(t0)
#     + "_spectrum_"
#     + str(np.count_nonzero(mask_object == 0))
#     + self.projectionorder[which_proj]
#     + ".svg"
# )
