sudo docker run -it --privileged\
        --net=host\
        -v /dev:/dev\
        -v /run/udev:/run/udev\
        --group-add video\
        -v /home/nautilus/codice:/home/ubuntu\
        ghcr.io/nautilus-unipd/jetson-nano-setup:latest

