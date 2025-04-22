# clara-viz 0.4.1 (April 16 2025)

## Features

* Add support CMake 4.0
* Add support for Python 3.12 packages, remove 3.6, 3.7 and 3.8 which are end of life

## Bug Fixes

* AlmaLinux 8 GPG key update (https://almalinux.org/blog/2023-12-20-almalinux-8-key-update/)

## Security

* Update packages to fix vulnerabilities

# clara-viz 0.4.0 (April 10 2024)

## Features

* Add Python and pybind11 source code
* Add support for Python 3.10 and 3.11 packages and set Python wheel requirement to fail installation on unsupported Python versions

## Bug Fixes

* Add missing <thread> include to DataSourceUS.cpp ([32](https://github.com/NVIDIA/clara-viz/pull/32)) [@thewtex](https://github.com/thewtex)
* Upgrade almalinux-release in Docker build for GPG keys ([31](https://github.com/NVIDIA/clara-viz/pull/32)) [@thewtex](https://github.com/thewtex)
* Gracefully disable OptiX and NvENC features if initialization fails.
* Python: Fix segfault when accessing RGBA data of rendered image ([33](https://github.com/NVIDIA/clara-viz/issues/33)) [@thewtex](https://github.com/thewtex)

# clara-viz 0.3.2 (November 27 2023)

## Features

* Support camera pose for non-stereo cases as well

## Bug Fixes

* Fix cmake warnings and consolidate cmake_minimum_required() usage
* Fix compile warning in NvEnvService.cpp

# clara-viz 0.3.1 (July 12 2023)

## Bug Fixes

* Gracefully disable OptiX denoiser if it can't be created
* make nvjpeg support optional

# clara-viz 0.3.0 (June 22 2023)

## Features

* C++ interface is now public
* Add support for aarch64 (not for Python wheels)
* Add support for VR/AR rendering
  + Camera has stereo mode and left end right eye poses, tangents and gaze directions
  + Add warped and foveated rendering
  + Add reprojection
* Add support for separate depth buffer
* Add support for alpha channel
* Add support for near and far depth plane
* Add support for transforming the volume using a matrix

## Bug Fixes

* Fix tonemap enable, had no effect before (always enabled).
* AttributeError: module 'itk' has no attribute 'itkSpatialOrientationPython' (https://github.com/NVIDIA/clara-viz/issues/26)

## Security

* Update Jupyter widget Java code packages to fix vulnerability CVE-2022-46175

# clara-viz 0.2.2 (October 17 2022)

## Bug Fixes

* Jupyter notebooks fail with AttributeError: 'Widget' object has no attribute 'on_displayed' (https://github.com/NVIDIA/clara-viz/issues/23)

## Security

* Update Jupyter widget Java code packages to fix vulnerabilities

## Documentation

* pip install fails on Windows machines (https://github.com/NVIDIA/clara-viz/issues/22)
  * Windows is not supported, added a section on supported OS to readme

# clara-viz 0.2.1 (March 29 2022)

## Bug Fixes

* Widget can't be displayed because of version mismatch

# clara-viz 0.2.0 (March 29 2022)

## Features

* Add support for rendering multi resolution images used in digital pathology

## Security

* Update Jupyter widget Java code packages to fix vulnerabilities

## Bug Fixes

* Error when using a widget with a renderer using a numpy array (https://github.com/NVIDIA/clara-viz/issues/18)

## Documentation

* Fix typo for `image_type` parameter in the sample code of the readme file
* Extended documentation, added multi resolution image rendering

# clara-viz 0.1.4 (Feb 15 2022)

## Security

* Update Jupyter widget Java code packages to fix vulnerabilities

## Bug Fixes

* Regression - cinematic rendering example throws an error (https://github.com/NVIDIA/clara-viz/issues/16)

# clara-viz 0.1.3 (Jan 31 2022)

## New

* Support installation of recommended dependencies

## Bug Fixes

* Failed to load data files with ITK when using Clara Train docker image (https://github.com/NVIDIA/clara-viz/issues/12)
* Rendering is wrong when passing non-contiguous data in (e.g. transposed numpy array)
* Widget interaction in slice mode not working correctly (https://github.com/NVIDIA/clara-viz/issues/13)

# clara-viz 0.1.2 (Jan 19 2022)

## Bug Fixes

* When the renderer is immediately destroyed after creation there is a segmentation fault. This could happen when providing a unsupported data type (e.g. 64 bit floating point values), when creating a temporary object (e.g. in Python `print(Renderer(data)))`) or when the initialization of the Renderer failed. (https://github.com/NVIDIA/clara-viz/issues/7, https://github.com/NVIDIA/clara-viz/issues/8)
* Widget is not working with Jupyter Notebooks (but with Jupyter Lab) (https://github.com/NVIDIA/clara-viz/issues/9)

## Documentation

* Added missing `video` capability to docker run command

# clara-viz 0.1.1 (Dec 14 2021)

## Bug Fixes

* When installing the `clara-viz-core` Python package only there is the error `ModuleNotFoundError: No module named 'packaging'` when doing `import clara.viz.core`
* When getting the settings from the renderer the 'TransferFunctions' sections is returned as 'Transferfunctions' with lower case 'f'

## Documentation

* Added a section on using Clara Viz within a docker container.
* Added a link to the documentation.
* Added a section on WSL (Windows Subsystem for Linux).

## Notebooks

* The DataDefinition class is using ITK to load the data files, make sure ITK is available.
* Added a slice rendering example (Slice_rendering.ipynb)
* Fixed the check if the volume file exists in Render_image.ipynb, also fixed volume orientation and scaling.
* Updated the settings files to match the settings conventions used by the renderer.

## Misc

* Changed the camera names and removed the `Slice` prefix of the orthographic cameras. Renamed the perspective camera from `Cinematic` to `Perspective`

# clara-viz 0.1.0 (Dec 3 2021)

Initial release of Clara Viz
