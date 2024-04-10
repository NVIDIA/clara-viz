# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Module version
from packaging import version

# _version.py is used both by setup.py and __init__.py. setup.py is using
# exec() from the parent folder and __init__.py is using import from the
# current folder. We have to look at different locations for the VERSION
# file.
try:
    __file__
except:
    with open('clara/viz/core/VERSION', 'r') as f:
        version = version.parse(f.read())
else:
    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, 'VERSION'), 'r') as f:
        version = version.parse(f.read())

version_info = (version.major, version.minor, version.micro, 'final', 0)

# Module version stage suffix map
_specifier_ = {'alpha': 'a', 'beta': 'b', 'candidate': 'rc', 'final': ''}

# Module version accessible using clara.viz.__version__
__version__ = '%s.%s.%s%s' % (version_info[0], version_info[1], version_info[2],
                              '' if version_info[3] == 'final' else _specifier_[version_info[3]]+str(version_info[4]))
