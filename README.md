# Clara Viz

NVIDIA Clara Viz is a platform for visualization of 2D/3D medical imaging data. It enables building applications
that leverage powerful volumetric visualization using CUDA-based ray tracing. It also allows viewing of multi resolution
images used in digital pathology.

<div style="display: flex; width: 100%; justify-content: center;">
  <div style="padding: 5px; height: 200px;">
    <img src="images/rendering.gif" alt="Volume Rendering"/>
  </div>
  <div style="padding: 5px; height: 200px;">
    <img src="images/pathology.gif" alt="Pathology"/>
 </div>
</div>

Clara Viz offers a Python Wrapper for rapid experimentation. It also includes a collection of
visual widgets for performing interactive medical image visualization in Jupyter Lab notebooks.

## Known issues

On Windows, starting with Chrome version 91 (also with Microsoft Edge) the interactive Jupyter widget is not working correctly. There is a delay in the interactive view after starting interaction. This is an issue with the default (D3D11) rendering backend of the browser. To fix this open `chrome://flags/#use-angle` and switch the backend to `OpenGL`.

## Requirements

* OS: Linux x86_64 or aarch64
* NVIDIA GPU: Pascal or newer, including Pascal, Volta, Turing and Ampere families
* NVIDIA driver: 450.36.06+

## Documentation

https://docs.nvidia.com/clara-viz/index.html

## Build

### With docker file

This is using a docker file to build the binaries. First build the docker file used to compile the code:

```bash
docker build -t clara_viz_builder_$(uname -m) -f Dockerfile_$(uname -m).build .
```

Then start the build process inside the build docker image. Build results are written to the 'build' directory.

```bash
docker run --network host --rm -it -u $(id -u):$(id -g) -v $PWD:/ClaraViz \
    -w /ClaraViz clara_viz_builder_$(uname -m) ./build.sh -o build_$(uname -m)
```

### From command line

#### Dependencies

git
git-lfs
nasm
CMake 3.19.1

#### Build

```bash
./build.sh -o build_$(uname -m)
```

## Use within a Docker container

Clara Viz requires CUDA, use a `base` container from `https://hub.docker.com/r/nvidia/cuda` for example `nvidia/cuda:11.4.2-base-ubuntu20.04`. By default the CUDA container exposes the `compute` and `utility` capabilities only. Clara Viz additionally needs the `graphics` and `video` capabilities. Therefore the docker container needs to be run with the `NVIDIA_DRIVER_CAPABILITIES` env variable set:
```bash
$ docker run -it --rm -e NVIDIA_DRIVER_CAPABILITIES=graphics,video,compute,utility nvidia/cuda:11.4.2-base-ubuntu20.04
```
or add:
```
ENV NVIDIA_DRIVER_CAPABILITIES graphics,video,compute,utility
```
to your docker build file.
See https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html#driver-capabilities for more information.

## WSL (Windows Subsystem for Linux)

Currently Clara Viz won't run under WSL because OptiX is not supported in that environment.

## Acknowledgments

Without awesome third-party open source software, this project wouldn't exist.

Please find `LICENSE-3rdparty.md` to see which third-party open source software
is used in this project.

## License

Apache-2.0 License (see `LICENSE` file).

Copyright (c) 2020-2023, NVIDIA CORPORATION.
