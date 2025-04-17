# Copyright (c) 2020-2025, NVIDIA CORPORATION. All rights reserved.
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

from __future__ import print_function
from setuptools import setup, find_namespace_packages
import os
import io
from distutils import log

from jupyter_packaging import (
    create_cmdclass,
    install_npm,
    ensure_targets,
    combine_commands,
)


here = os.path.dirname(os.path.abspath(__file__))

log.set_verbosity(log.DEBUG)
log.info('setup.py entered')
log.info('$PATH=%s' % os.environ['PATH'])


def read(*names, **kwargs):
    with io.open(
        os.path.join(here, *names),
        encoding=kwargs.get('encoding', 'utf8')
    ) as fh:
        return fh.read()


# Get clara-viz version
version = {}
with open(os.path.join(here, 'clara', 'viz', 'widgets', '_version.py')) as f:
    exec(f.read(), {}, version)

js_dir = os.path.join(here, 'js')

# Representative files that should exist after a successful build
jstargets = [
    os.path.join(js_dir, 'dist', 'index.js'),
]

data_files_spec = [
    ('share/jupyter/nbextensions/clara-viz-widgets', 'clara-viz-widgets/nbextension', '*.*'),
    ('share/jupyter/labextensions/clara-viz-widgets', 'clara-viz-widgets/labextension', '**'),
    ('share/jupyter/labextensions/clara-viz-widgets', '.', 'install.json'),
    ('etc/jupyter/nbconfig/notebook.d', '.', 'clara-viz-widgets.json'),
]

cmdclass = create_cmdclass('jsdeps', data_files_spec=data_files_spec)
cmdclass['jsdeps'] = combine_commands(
    install_npm(js_dir, npm=['yarn'], build_cmd='build:prod'), ensure_targets(jstargets),
)

setup_args = dict(
    name='clara-viz-widgets',
    version=version['__version__'],
    description='A toolkit to provide GPU accelerated visualization of medical data.',
    long_description='%s\n%s' % (
        read('README.md'),
        read('CHANGELOG.md')
    ),
    long_description_content_type='text/markdown',
    license='Apache-2.0',
    install_requires=[
        'packaging',
        'clara-viz-core==' + version['__version__'],
        'ipywidgets>=7.6.0,<8.0'
    ],
    packages=find_namespace_packages(include=["clara.*"]),
    include_package_data=True,
    zip_safe=False,
    cmdclass=cmdclass,
    author='NVIDIA Corporation',
    url='https://github.com/NVIDIA/clara-viz',
    keywords=[
        'ipython',
        'jupyter',
        'widgets',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Framework :: IPython',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Multimedia :: Graphics',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    python_requires='>= 3.9, < 3.13',
    platforms=['manylinux_2_28_x86_64'],
)

setup(**setup_args)
