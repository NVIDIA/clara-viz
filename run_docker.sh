# build before with docker build -t clara-viz-jlab .

docker run \
    -it \
    --rm \
    -p 8099:8888 \
    --name clara-viz \
    --volume $(pwd):/home/jovyan/work \
    --volume "/media/t/Seagate Expansion Drive/UKA/imageData/clara-viz/input/images":/workspace/inputs \
    --volume "/media/t/Seagate Expansion Drive/UKA/imageData/clara-viz/results":/workspace/results \
    -e NVIDIA_DRIVER_CAPABILITIES=graphics,video,compute,utility \
    clara-viz-jlab