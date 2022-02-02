# build before with docker build -t claraviz:latest . or Dockerfile right-click in VSCode -> build <3

docker run -it --rm \
    --gpus=all \
    -p 8888:8888 \
    --name clara-viz \
    --volume $(pwd):/home/jovyan/work \
    -e NVIDIA_DRIVER_CAPABILITIES=graphics,video,compute,utility \
    claraviz:latest