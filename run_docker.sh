# build before with docker build -t claraviz:latest . or Dockerfile right-click in VSCode -> build <3

docker run \
    -it \
    --rm \
    --gpus=all \
    -p 8888:8888 \
    --name clara-viz \
    --volume $(pwd):/home/jovyan/work \
    --volume "/media/t/Seagate Expansion Drive/UKA/imageData/clara-viz/input/images":/workspace/inputs \ #not necessary
    --volume "/media/t/Seagate Expansion Drive/UKA/imageData/clara-viz/results":/workspace/results \ #not necessary
    -e NVIDIA_DRIVER_CAPABILITIES=graphics,video,compute,utility \
    claraviz:latest