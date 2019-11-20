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
from builtins import range
import re
import h5py
import numpy
import vigra
from ilastik.applets.base.appletSerializer import (
    AppletSerializer,
    SerialClassifierSlot,
    SerialBlockSlot,
    SerialListSlot,
    SerialClassifierFactorySlot,
    SerialPickleableSlot,
)
from lazyflow.slot import OutputSlot
from typing import List, Tuple
from ndstructs import Array5D, Slice5D

import logging

logger = logging.getLogger(__name__)


class BackwardsCompatibleLabelSerialBlockSlot(SerialBlockSlot):
    def get_corresponding_input_image_slot(self, labelSlot: OutputSlot) -> OutputSlot:
        labelMultislot = labelSlot.operator.inputs[labelSlot.name]
        for subslotIndex in range(len(labelMultislot)):
            if labelMultislot[subslotIndex] == labelSlot:
                return labelSlot.operator.InputImages[subslotIndex]

    def get_input_image_original_axiskeys(self, labelSlot: OutputSlot) -> str:
        return "".join(self.get_corresponding_input_image_slot(labelSlot).meta.getOriginalAxisKeys())

    def get_input_image_current_axiskeys(self, labelSlot: OutputSlot) -> str:
        return "".join(self.get_corresponding_input_image_slot(labelSlot).meta.getAxisKeys())

    def reshape_datablock_and_slicing_for_input(
        self, block: numpy.ndarray, slicing: List[slice], slot: OutputSlot, project_file: h5py.File
    ) -> Tuple[numpy.ndarray, List[slice]]:
        """Reshapes a block of data and its corresponding slicing into the slot's current shape, so as to be
        compatible with versions of ilastik that saved and loaded block slots in their original shape

        Checks for version 1.3.3 and 1.3.3post1 because those were the versions that saved labels in 5D
        """
        project_file_version = project_file["/ilastikVersion"][()].decode("utf-8")
        project_version_parts = re.compile(r"\.|post").split(project_file_version)
        numeric_version = tuple(int(part) for part in project_version_parts)
        workflow_name = project_file["/workflowName"][()].decode("utf-8")
        pixel_plus_object_workflow_name = "Object Classification (from pixel classification)"

        current_axiskeys = self.get_input_image_current_axiskeys(slot)
        if (1, 3, 3) <= numeric_version < (1, 3, 3, 2) and workflow_name == pixel_plus_object_workflow_name:
            saved_data_axiskeys = current_axiskeys
            self.dirty = True
        else:
            saved_data_axiskeys = self.get_input_image_original_axiskeys(slot)

        fixed_slicing = Slice5D.zero(**dict(zip(saved_data_axiskeys, slicing))).to_slices(current_axiskeys)
        fixed_block = Array5D(block, saved_data_axiskeys).raw(current_axiskeys)
        return fixed_block, fixed_slicing

    def reshape_datablock_and_slicing_for_output(
        self, block: numpy.ndarray, slicing: List[slice], slot: OutputSlot
    ) -> Tuple[numpy.ndarray, List[slice]]:
        """Reshapes a block of data and its corresponding slicing into the slot's original shape, so as to be
        compatible with versions of ilastik that saved and loaded block slots in their original shape

        Always save using original shape to be backwards compatible with 1.3.2
        """
        original_axiskeys = self.get_input_image_original_axiskeys(slot)
        current_axiskeys = self.get_input_image_current_axiskeys(slot)
        fixed_block = Array5D(block, current_axiskeys).raw(original_axiskeys)
        fixed_slicing = Slice5D.zero(**dict(zip(current_axiskeys, slicing))).to_slices(original_axiskeys)
        return fixed_block, fixed_slicing


class PixelClassificationSerializer(AppletSerializer):
    """Encapsulate the serialization scheme for pixel classification
    workflow parameters and datasets.

    """

    def __init__(self, operator, projectFileGroupName):
        self.VERSION = 1
        self._serialClassifierSlot = SerialClassifierSlot(
            operator.Classifier, operator.classifier_cache, name="ClassifierForests"
        )
        slots = [
            SerialListSlot(operator.LabelNames),
            SerialListSlot(operator.LabelColors, transform=lambda x: tuple(x.flat)),
            SerialListSlot(operator.PmapColors, transform=lambda x: tuple(x.flat)),
            SerialPickleableSlot(operator.Bookmarks, self.VERSION),
            BackwardsCompatibleLabelSerialBlockSlot(
                operator.LabelImages,
                operator.LabelInputs,
                operator.NonzeroLabelBlocks,
                name="LabelSets",
                subname="labels{:03d}",
                selfdepends=False,
                shrink_to_bb=True,
            ),
            SerialClassifierFactorySlot(operator.ClassifierFactory),
            self._serialClassifierSlot,
        ]

        super(PixelClassificationSerializer, self).__init__(projectFileGroupName, slots, operator)

    def _deserializeFromHdf5(self, topGroup, groupVersion, hdf5File, projectFilePath, headless=False):
        """
        Override from AppletSerializer.
        Implement any additional deserialization that wasn't already accomplished by our list of serializable slots.
        """
        # If this is an old project file that didn't save the label names to the project,
        #   create some default names.
        if (
            not self.operator.LabelNames.ready() or len(self.operator.LabelNames.value) == 0
        ) and "LabelSets" in topGroup:
            # How many labels are there?
            # We have to count them.
            # This is slow, but okay for this special backwards-compatibilty scenario.

            # For each image
            all_labels = set()
            for image_index, group in enumerate(topGroup["LabelSets"].values()):
                # For each label block
                for block in list(group.values()):
                    data = block[:]
                    all_labels.update(vigra.analysis.unique(data))

            if all_labels:
                max_label = max(all_labels)
            else:
                max_label = 0

            label_names = []
            for i in range(max_label):
                label_names.append("Label {}".format(i + 1))

            self.operator.LabelNames.setValue(label_names)
            # Make some default colors, too
            default_colors = [
                (255, 0, 0),
                (0, 255, 0),
                (0, 0, 255),
                (255, 255, 0),
                (255, 0, 255),
                (0, 255, 255),
                (128, 128, 128),
                (255, 105, 180),
                (255, 165, 0),
                (240, 230, 140),
            ]
            colors = []
            for i, _ in enumerate(label_names):
                colors.append(default_colors[i])
            self.operator.LabelColors.setValue(colors)
            self.operator.PmapColors.setValue(colors)

            # Now RE-deserialize the classifier, so it isn't marked dirty
            self._serialClassifierSlot.deserialize(topGroup)

        # SPECIAL CLEANUP for backwards compatibility:
        # Due to a bug, it was possible for a project to be saved with a classifier that was
        #  trained with more label classes than the project file saved in the end.
        # That can cause a crash.  So here, we inspect the restored classifier and remove it if necessary.
        if not self.operator.classifier_cache._dirty:
            restored_classifier = self.operator.classifier_cache._value
            if hasattr(restored_classifier, "known_classes"):
                num_classifier_classes = len(restored_classifier.known_classes)
                num_saved_label_classes = len(self.operator.LabelNames.value)
                if num_classifier_classes > num_saved_label_classes:
                    # Delete the classifier from the operator
                    logger.info("Resetting classifier... will be forced to retrain")
                    self.operator.classifier_cache.resetValue()


class Ilastik05ImportDeserializer(AppletSerializer):
    """
    Special (de)serializer for importing ilastik 0.5 projects.
    For now, this class is import-only.  Only the deserialize function is implemented.
    If the project is not an ilastik0.5 project, this serializer does nothing.
    """

    def __init__(self, topLevelOperator):
        super(Ilastik05ImportDeserializer, self).__init__("")
        self.mainOperator = topLevelOperator

    def serializeToHdf5(self, hdf5Group, projectFilePath):
        """Not implemented. (See above.)"""
        pass

    def deserializeFromHdf5(self, hdf5File, projectFilePath, headless=False):
        """If (and only if) the given hdf5Group is the root-level group of an
           ilastik 0.5 project, then the project is imported.  The pipeline is updated
           with the saved parameters and datasets."""
        # The group we were given is the root (file).
        # Check the version
        ilastikVersion = hdf5File["ilastikVersion"].value

        # The pixel classification workflow supports importing projects in the old 0.5 format
        if ilastikVersion == 0.5:
            numImages = len(hdf5File["DataSets"])
            self.mainOperator.LabelInputs.resize(numImages)

            if numImages == 0:
                return

            first_group_name, first_group = sorted(hdf5File["DataSets"].items())[0]
            label_names = first_group["labels"].attrs["name"]
            label_hexcolors = first_group["labels"].attrs["color"]

            color_tuples = []
            for color_hex in label_hexcolors:
                red = (color_hex & 0xFF0000) >> 16
                green = (color_hex & 0x00FF00) >> 8
                blue = (color_hex & 0x0000FF) >> 0
                color_tuples.append((red, green, blue))

            self.mainOperator.LabelNames.setValue(label_names)
            self.mainOperator.LabelColors.setValue(color_tuples)
            self.mainOperator.PmapColors.setValue(color_tuples)

            for index, (datasetName, datasetGroup) in enumerate(sorted(hdf5File["DataSets"].items())):
                try:
                    dataset = datasetGroup["labels/data"]
                except KeyError:
                    # We'll get a KeyError if this project doesn't have labels for this dataset.
                    # That's allowed, so we simply continue.
                    pass
                else:
                    slicing = [slice(0, s) for s in dataset.shape]
                    self.mainOperator.LabelInputs[index][slicing] = dataset[...]

    def importClassifier(self, hdf5File):
        """
        Import the random forest classifier (if any) from the v0.5 project file.
        """
        # Not yet implemented.
        # The old version of ilastik didn't actually deserialize the
        #  classifier, but it did determine how many trees were used.
        pass

    def isDirty(self):
        """Always returns False because we don't support saving to ilastik0.5 projects"""
        return False

    def unload(self):
        # This is a special-case import deserializer.  Let the real deserializer handle unloading.
        pass

    def _serializeToHdf5(self, topGroup, hdf5File, projectFilePath):
        assert False

    def _deserializeFromHdf5(self, topGroup, groupVersion, hdf5File, projectFilePath, headless=False):
        # This deserializer is a special-case.
        # It doesn't make use of the serializer base class, which makes assumptions about the file structure.
        # Instead, if overrides the public serialize/deserialize functions directly
        assert False
