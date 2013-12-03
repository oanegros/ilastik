import utility
import roi
import rtype
import stype
import operators
import request
import graph
import slot

import os
this_file = os.path.abspath(__file__)
lazyflow_package_dir = os.path.dirname(this_file)
lazyflow_repo_dir = os.path.dirname(lazyflow_package_dir)
submodule_dir = os.path.join( lazyflow_repo_dir, 'submodules' )

# Add all submodules to the PYTHONPATH
import expose_submodules
expose_submodules.expose_submodules(submodule_dir)
