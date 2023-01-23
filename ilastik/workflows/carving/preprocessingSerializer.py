from __future__ import absolute_import

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
###############################################################################
from ilastik.applets.base.appletSerializer import AppletSerializer, getOrCreateGroup, deleteIfPresent
import h5py
import numpy
import os

from .watershed_segmentor import WatershedSegmentor


class PreprocessingSerializer(AppletSerializer):
    def __init__(self, preprocessingTopLevelOperator, *args, **kwargs):
        super(PreprocessingSerializer, self).__init__(*args, **kwargs)
        self._o = preprocessingTopLevelOperator

    def _serializeToHdf5(self, topGroup, hdf5File, projectFilePath):
        preproc = topGroup

        for opPre in self._o.innerOperators:
            mst = opPre._prepData[0]

            if mst is not None:

                # The values to be saved for sigma and filter are the
                # values of the last valid preprocess
                #!These may differ from the current settings!

                deleteIfPresent(preproc, "sigma")
                deleteIfPresent(preproc, "filter")
                deleteIfPresent(preproc, "do_agglomeration")
                deleteIfPresent(preproc, "size_regularizer")
                deleteIfPresent(preproc, "reduce_to")
                deleteIfPresent(preproc, "graph")

                preproc.create_dataset("sigma", data=opPre.initialSigma)
                preproc.create_dataset("filter", data=opPre.initialFilter)
                preproc.create_dataset("do_agglomeration", data=opPre.initialDoAgglo)
                preproc.create_dataset("size_regularizer", data=opPre.initialSizeRegularizer)
                preproc.create_dataset("reduce_to", data=opPre.initialReduceTo)

                preprocgraph = getOrCreateGroup(preproc, "graph")
                mst.saveH5G(preprocgraph)

            opPre._unsavedData = False

    def _deserializeFromHdf5(self, topGroup, groupVersion, hdf5File, projectFilePath, headless=False):

        assert "sigma" in list(topGroup.keys())
        assert "filter" in list(topGroup.keys())

        sigma = topGroup["sigma"][()]
        sfilter = topGroup["filter"][()]

        # Pre-1.4.0rc9 project files do not contain doAgglo, sizeRegularizer and reduceTo - use defaults
        doAgglo = topGroup["do_agglomeration"][()] if "do_agglomeration" in list(topGroup.keys()) else 1
        sizeRegularizer = topGroup["size_regularizer"][()] if "size_regularizer" in list(topGroup.keys()) else 0.5
        reduceTo = topGroup["reduce_to"][()] if "reduce_to" in list(topGroup.keys()) else 0.2

        if "graph" in list(topGroup.keys()):
            graphgroup = topGroup["graph"]
        else:
            assert "graphfile" in list(topGroup.keys())
            # feature: load preprocessed graph from file
            filePath = topGroup["graphfile"][()]
            if not os.path.exists(filePath):
                if headless:
                    raise RuntimeError("Could not find data at " + filePath)
                filePath = self.repairFile(filePath, "*.h5")
            graphgroup = h5py.File(filePath, "r")["graph"]

        for opPre in self._o.innerOperators:

            opPre.initialSigma = sigma
            opPre.initialFilter = sfilter
            opPre.initialDoAgglo = doAgglo
            opPre.initialSizeRegularizer = sizeRegularizer
            opPre.initialReduceTo = reduceTo

            opPre.Sigma.setValue(sigma)
            opPre.Filter.setValue(sfilter)
            opPre.DoAgglo.setValue(doAgglo)
            opPre.SizeRegularizer.setValue(sizeRegularizer)
            opPre.ReduceTo.setValue(reduceTo)

            mst = WatershedSegmentor(h5file=graphgroup)
            opPre._prepData = numpy.array([mst])

            opPre._dirty = False
            opPre.applet.writeprotected = True

            opPre.PreprocessedData.setDirty()
            opPre.enableDownstream(True)

    def isDirty(self):
        for opPre in self._o.innerOperators:
            if opPre._unsavedData:
                return True
        return False

    # this is present only for the serializer AppletInterface
    def unload(self):
        pass
