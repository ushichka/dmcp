sudo podman run --privileged --device nvidia.com/gpu=all -v $(pwd):/workspace:rw -p 8888:8888 --shm-size=8g --rm -it local/hloc bash 


