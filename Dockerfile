FROM ros:jazzy

USER root

# Create, if not existing, 'ubuntu' user
RUN id -u ubuntu &>/dev/null || useradd -m -s /bin/bash ubuntu

# Set no password to 'ubuntu'
RUN apt-get update && apt-get install -y sudo && \
    echo 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/ubuntu-nopasswd && \
    chmod 0440 /etc/sudoers.d/ubuntu-nopasswd

# Install base dependencies
RUN apt-get update && \
    apt-get install -y python3-pip git python3-jinja2 python3-colcon-meson \
    libboost-dev libgnutls28-dev openssl libtiff5-dev pybind11-dev \
    qtbase5-dev libqt5core5a libqt5gui5 libqt5widgets5 meson cmake \
    python3-yaml python3-ply libglib2.0-dev libgstreamer-plugins-base1.0-dev \
    libboost-program-options-dev libdrm-dev libexif-dev ninja-build \
    libpng-dev libopencv-dev libavdevice-dev libepoxy-dev \
    ros-jazzy-cv-bridge python3-opencv


# Set environment for the ubuntu user
ENV HOME=/home/ubuntu

# Set up .bashrc for ubuntu
RUN echo "cd \$HOME" >> /home/ubuntu/.bashrc && \
    echo "source /opt/ros/jazzy/local_setup.bash" >> /home/ubuntu/.bashrc && \
    chown ubuntu:ubuntu /home/ubuntu/.bashrc

# Set up default display
ENV DISPLAY=:0

# Switch to the ubuntu user
USER ubuntu
WORKDIR /home/ubuntu
