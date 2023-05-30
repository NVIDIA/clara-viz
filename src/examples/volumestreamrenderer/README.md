# Clara Viz Volume Stream Render Server Example

This sample show how to use the Renderer for visualizing streaming
data efficently. I supports 3D volume data or 2D ultrasound data for
input. If no input data is specified then synthetical data is generated.

The example contains a docker compose definition which starts up all
needed containers.

There are three Docker containers

-   The Render Server
-   A web server
-   A proxy using Envoy translating between grpc-web and gRPC

See [gRPC-Web](https://github.com/grpc/grpc-web) for more information
on how to use gRPC from a browser client.

## Running

-   Install [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
-   Install [Docker Compose](https://docs.docker.com/compose/install)

First build the SDK. See the top level [README.md](../../../README.md)
for more information. Then build the Docker containers.

```shell
docker compose build
```

Finally start the Docker containers:

```shell
docker compose up
```

Open Google Chrome or Firefox, navigate to `http:://localhost:8081`.

This will show a rendering of a synthetic 3D CT volume sequence.

For rendering the ultrasound sequence or rendering own datasets pass
arguments to the `renderserver` container. Assuming docker compose is
not running first start the web service containers.

```shell
docker compose up --no-deps envoy web-client
```

Then start `renderserver` container with arguments:

```shell
docker compose run renderserver -s US
```

This will start the `renderserver` using the ultrasound scenario. Open the
browser at `http:://localhost:8081` to view the rendering. The
`renderseerver` container can be stopped and restarted with other arguments.

### Using CT datasets

Provide a directory with Volumes stored in `MHD` format. The files in that directory will be loaded in alphabetical order
and used for the sequence.
The Docker Compose configuration by default maps the local `data` directory into the container, so place your files in a local `data` folder

```shell
docker compose run renderserver -s CT -i data/my_data_directory
```

### Using US datasets

Provide a directory with raw slices stored and a file `PhiData.txt`
with information on the slices. The file `PhiData.txt` is a text
file containing the angle `PHI`, the time stamp, the width, the
depth, the scan depth and the x and y origin of the slice that had
been aquired. The file name of the raw slice data file is build
from the timestamp by adding the `.bin` extension.

For example this text snippet

```
...
0.76 1245 690 1116 120 0 0
0.79 1247 690 1116 120 0 0
...
```

specifies two slices `1245.bin` and `1247.bin`.

The files loaded in timestamp order and used for the sequence.
The Docker Compose configuration by default maps the local `data` directory into the container, so place your files in a local `data` folder.

```shell
docker compose run renderserver -s US -i data/my_data_directory
```

### Running without using a Docker container

You could even start the renderer without a container when for example using
your own dataset to be visualized:

```shell
./bin/$(uname -m)/VolumeStreamRenderer --input myDatasetDirectory
```

## Architecture

The web app runs in the browser and uses gRPC-web to communicate with
the Render Server. Then Envoy proxy translates the gRPC-web messages
rom the browser to gRPC messages on the backend.

### Render Server

The example Render Server uses the services provided by the Render Server SDK.

If the `CT` scenario is selected the server loads a sequence of 3D volume files
into GPU memory or generates synthetic 3D volumes and pushes them to the renderer at a fixed rate.

In the `US` scenario the server load 2D slices or generates synthetic data and uses
a back-projection algorithm to build a 3D volume which is pushed to the renderer. The
amount of 2D slices used is increased to show how the 3D volume builds up.

## Known issues

-   <b>When opening the browser at `http:://localhost:8081` no rendering is visible</b>
    <br>Sometimes the stream is not started when first opening the web-page. Try refreshing the browser window.
