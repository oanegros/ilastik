from __future__ import print_function

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
#           http://ilastik.org/license.html
###############################################################################
import os
import sys
import numpy as np
import h5py
import pytest

from lazyflow.utility.timer import timeLogged

import logging

logger = logging.getLogger(__name__)

SOLVER = None
try:
    import multiHypoTracking_with_cplex as mht

    SOLVER = "CPLEX"
except ImportError:
    try:
        import multiHypoTracking_with_gurobi as mht

        SOLVER = "GUROBI"
    except ImportError:
        logger.info("Could not find any ILP solver.")


@pytest.mark.skipif(SOLVER is None, reason="Could not find any ILP solver - unable to run learning tests!")
class TestStructuredLearningTrackingHeadless(object):

    logger.info("looking for tests directory ...")
    input_data_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "inputdata", "cell_tracking_challenge_15", "Fluo-N2DH-SIM"
    )
    if not os.path.exists(input_data_path):
        raise RuntimeError("Couldn't find ilastik/tests directory: {}".format(input_data_path))

    PROJECT_FILE = os.path.join(input_data_path, "01/learning-with-segmentation-gt-2017-01-17.ilp")
    RAW_DATA_FILE = os.path.join(input_data_path, "01/learningRaw-2017-01-17.h5")
    BINARY_SEGMENTATION_FILE = os.path.join(input_data_path, "01_GT/SEG/learningSeg-2017-01-17.h5")

    EXPECTED_TRACKING_RESULT_FILE = os.path.join(
        input_data_path, "01/learningRaw-2017-01-17-exported_data_Tracking-Result.h5"
    )
    EXPECTED_CSV_FILE = os.path.join(input_data_path, "01/learningRaw-2017-01-17_CSV-Table.csv")
    EXPECTED_SHAPE = (10, 495, 534, 1)  # Expected shape for tracking results HDF5 files
    EXPECTED_NUM_LINES_TRACKING = 164  # Number of lines expected in exported csv file
    EXPECTED_NUM_DIVISIONS = 6  # Number of lines expected in exported csv file
    EXPECTED_MERGER_NUM = 0  # Number of mergers expected in exported csv file
    EXPECTED_FALSE_DETECTIONS_NUM = 1  # Number of false detections expected in exported csv file

    @classmethod
    def setup_class(cls):
        if (
            not os.path.isfile(cls.PROJECT_FILE)
            or not os.path.isfile(cls.RAW_DATA_FILE)
            or not os.path.isfile(cls.BINARY_SEGMENTATION_FILE)
        ):
            raise RuntimeError("Test input files not found.")

        logger.info("starting setup...")
        cls.original_cwd = os.getcwd()

        # Load the ilastik startup script as a module.
        # Do it here in setupClass to ensure that it isn't loaded more than once.
        logger.info("looking for ilastik.py...")
        import ilastik.__main__

        cls.ilastik_startup = ilastik.__main__

    @classmethod
    def teardown_class(cls):
        removeFiles = [
            os.path.join(cls.input_data_path, "01/learningRaw-2017-01-17_Tracking-Result.h5"),
            os.path.join(cls.input_data_path, "01/learningRaw-2017-01-17_CSV-Table.csv"),
        ]

        # Clean up: Delete any test files we generated
        for f in removeFiles:
            try:
                os.remove(f)
            except:
                pass
        pass

    @timeLogged(logger)
    def testStructuredLearningTrackingHeadless(self):
        # Skip test if structured learning tracking can't be imported. If it fails the problem is most likely that CPLEX is not installed.
        try:
            import ilastik.workflows.tracking.structured
        except ImportError as e:
            pytest.xfail("Structured learning tracking could not be imported. CPLEX is most likely missing: " + str(e))

        args = " --project=" + self.PROJECT_FILE
        args += " --headless"
        args += " --export_source=Tracking-Result"
        args += " --raw_data " + self.RAW_DATA_FILE + "/exported_data"
        args += " --binary_image " + self.BINARY_SEGMENTATION_FILE + "/exported_data"

        sys.argv = ["ilastik.py"]  # Clear the existing commandline args so it looks like we're starting fresh.
        sys.argv += args.split()

        # Start up the ilastik.py entry script as if we had launched it from the command line
        self.ilastik_startup.main()

        # Examine the HDF5 output for basic attributes
        with h5py.File(self.EXPECTED_TRACKING_RESULT_FILE, "r") as f:
            assert "exported_data" in f, "Dataset does not exist in the tracking result file"
            data_shape = f["exported_data"].shape
            logger.info("Exported data shape: {}".format(data_shape))
            print("Exported data shape: {}".format(data_shape))
            assert data_shape == self.EXPECTED_SHAPE, "Exported data {} has a wrong shape: {}".format(
                self.EXPECTED_SHAPE, data_shape
            )

    @timeLogged(logger)
    def testCSVExport(self):
        # TODO: When Hytra is supported on Windows, we shouldn't skip the test and throw an assert instead
        try:
            import hytra
        except ImportError as e:
            pytest.xfail("Hytra tracking pipeline couldn't be imported: " + str(e))

        args = " --project=" + self.PROJECT_FILE
        args += " --headless"

        args += " --export_source=Plugin"
        args += " --export_plugin=CSV-Table"
        args += " --raw_data " + self.RAW_DATA_FILE  # + '/data'
        args += " --binary_image " + self.BINARY_SEGMENTATION_FILE + "/exported_data"

        sys.argv = ["ilastik.py"]  # Clear the existing commandline args so it looks like we're starting fresh.
        sys.argv += args.split()

        # Start up the ilastik.py entry script as if we had launched it from the command line
        self.ilastik_startup.main()

        # Load csv file
        data = np.genfromtxt(self.EXPECTED_CSV_FILE, dtype=float, delimiter=",", names=True)

        # Check for expected number of lines
        logger.info("Number of rows in the csv file: {}".format(data.shape[0]))
        print("Number of rows in the csv file: {}".format(data.shape[0]))
        assert data.shape[0] == self.EXPECTED_NUM_LINES_TRACKING, "Number of rows in the csv file differs from expected"

        # Check that the csv file contains the default fields.
        assert "frame" in data.dtype.names, "'frame' not found in the csv file!"
        assert "labelimageId" in data.dtype.names, "'labelimageId' not found in the csv file!"
        assert "lineageId" in data.dtype.names, "'lineageId' not found in the csv file!"
        assert "trackId" in data.dtype.names, "'trackId' not found in the csv file!"
        assert "parentTrackId" in data.dtype.names, "'parentTrackId' not found in the csv file!"
        assert "mergerLabelId" in data.dtype.names, "'mergerLabelId' not found in the csv file!"
        assert "Terminal_2_0" in data.dtype.names, "'Terminal_2_0' not found in the csv file!"
        assert "Terminal_2_1" in data.dtype.names, "'Terminal_2_1' not found in the csv file!"
        assert "Diameter_0" in data.dtype.names, "'Diameter_0' not found in the csv file!"
        assert "Bounding_Box_Minimum_0" in data.dtype.names, "'Bounding_Box_Minimum_0' not found in the csv file!"
        assert "Bounding_Box_Minimum_1" in data.dtype.names, "'Bounding_Box_Minimum_1' not found in the csv file!"
        assert "Center_of_the_object_0" in data.dtype.names, "'Center_of_the_object_0' not found in the csv file!"
        assert "Center_of_the_object_1" in data.dtype.names, "'Center_of_the_object_1' not found in the csv file!"
        assert "Bounding_Box_Maximum_0" in data.dtype.names, "'Bounding_Box_Maximum_0' not found in the csv file!"
        assert "Bounding_Box_Maximum_1" in data.dtype.names, "'Bounding_Box_Maximum_1' not found in the csv file!"

        # Check for expected number of mergers
        merger_count = 0
        previous = 0
        for id in data["mergerLabelId"]:
            if previous == 0 and not id == 0:
                merger_count += 1
            previous = id
        logger.info("Number of mergers in the csv file: {}".format(merger_count))
        assert (
            merger_count == self.EXPECTED_MERGER_NUM
        ), "Number of mergers {} in the csv file differs from expected {}.".format(
            merger_count, self.EXPECTED_MERGER_NUM
        )

        # Check for expected number of false detections
        false_detection_count = 0
        for id in data["lineageId"]:
            if id == -1:
                false_detection_count += 1
        logger.info("Number of false detections in the csv file: {}".format(false_detection_count))
        assert (
            false_detection_count == self.EXPECTED_FALSE_DETECTIONS_NUM
        ), "Number of false detections {} in the csv file differs from expected {}.".format(
            false_detection_count, self.EXPECTED_FALSE_DETECTIONS_NUM
        )

        # Check for expected number of divisions
        division_count = 0
        for id in data["parentTrackId"]:
            if not id == 0:
                division_count += 1
        division_count /= 2
        logger.info("Number of divisions in the csv file: {}".format(division_count))
        assert (
            division_count == self.EXPECTED_NUM_DIVISIONS
        ), "Number of divisions {} in the csv file differs from expected {}.".format(
            division_count, self.EXPECTED_NUM_DIVISIONS
        )
