make sure nvidia docker including container toolkit is installed properly.
execute xhost +local:docker / xhost +local:podman for gui support

Scripts use podman, with a slightly modified command to use nvidia gpus, this part needs to be modified for docker

Default password for jupyter set in some scripts is password
