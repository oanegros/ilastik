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
        print(image.shape)
        wavenames = ["wave_" + str(i + 1).zfill(3) for i in range(self.fineness)]
        result = {}
        for wavename in wavenames:
            result[wavename] = 1
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
        return self.combine_dicts(results)

    def compute_local(self, image, binary_bbox, features, axes):
        print("in compute local of spherical mip")
        margin = ilastik.applets.objectExtraction.opObjectExtraction.max_margin({"": features})
        passed, excl = ilastik.applets.objectExtraction.opObjectExtraction.make_bboxes(binary_bbox, margin)
        return self.do_channels(
            self._do_3d, image, label_bboxes=[binary_bbox, passed, excl], features=features, axes=axes
        )

    # def compute_global(self, image, labels, features, axes):
    #     return {}


# def _do_4d(self, image, labels, features, axes):

#     # ignoreLabel=None calculates background label parameters
#     # ignoreLabel=0 ignores calculation of background label parameters
#     assert isinstance(labels, vigra.VigraArray) and hasattr(labels, "axistags")
#     try:
#         result = vigra.analysis.extract3DConvexHullFeatures(labels.squeeze().astype(numpy.uint32), ignoreLabel=0)
#     except:
#         return dict()

#     # find the number of objects
#     try:
#         nobj = result[features[0]].shape[0]
#     except Exception as e:
#         logger.error(
#             "Feature name not found in computed features.\n"
#             "Your project file might be using obsolete features.\n"
#             "Please select new features, and re-train your classifier.\n"
#             "(Exception was: {})".format(e)
#         )
#         raise  # FIXME: Consider using Python 3 raise ... from ... syntax here.

#     # NOTE: this removes the background object!!!
#     # The background object is always present (even if there is no 0 label) and is always removed here
#     return cleanup(result, nobj, features)
