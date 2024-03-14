podman run --privileged --device nvidia.com/gpu=all -v $(pwd):/workspace:rw -p 8888:8888 --shm-size=8g --rm -it local/hloc jupyter lab --allow-root --no-browser --ip 0.0.0.0 / 


