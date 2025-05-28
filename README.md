# jetson-nano-setup

**Jetson Nano Setup Instructions**

This repository contains the Dockerfile and container image to run Ubuntu 24.04 (Noble) with ROS 2 Jazzy on a Jetson Nano.

---

## 1. Install Nvidia Jetpack

Follow NVIDIA’s official guide to flash your Jetson Nano with Jetpack 4.6.1: [https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#write](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#write)


Once complete, boot your Jetson Nano and perform any initial OS setup (username, locale, Wi‑Fi, etc.).

---

## 2. Configure VNC Server

Enable GNOME’s Vino VNC server to allow remote desktop access:

1. Open a terminal on the Jetson Nano:

   ```bash
   cd /usr/lib/systemd/user/graphical-session.target.wants
   sudo ln -s ../vino-server.service ./
   ```
2. Disable prompts and encryption, and require VNC authentication:

   ```bash
   gsettings set org.gnome.Vino prompt-enabled false
   gsettings set org.gnome.Vino require-encryption false
   gsettings set org.gnome.Vino authentication-methods "['vnc']"
   ```
3. Set a VNC password (replace `user` with your password):

   ```bash
   gsettings set org.gnome.Vino vnc-password $(echo -n 'user' | base64)
   ```
4. Reboot to apply changes:

   ```bash
   sudo reboot
   ```

Reference: [https://developer.nvidia.com/embedded/learn/tutorials/vnc-setup](https://developer.nvidia.com/embedded/learn/tutorials/vnc-setup)

---

## 3. Adjust X11 Configuration for Virtual Display

If you need a custom framebuffer resolution, append this section to `/etc/X11/xorg.conf`:

```bash
sudo bash -c 'cat >> /etc/X11/xorg.conf <<EOF
Section "Screen"
    Identifier "Default Screen"
    Monitor    "Configured Monitor"
    Device     "Tegra0"
    SubSection "Display"
        Depth    24
        Virtual  1280 800
    EndSubSection
EndSection
EOF'
```

This will create a virtual screen of 1280×800 at 24‑bit color depth.
Without this edit, if you access the Jetson only via VNC (without a physical display), the default resolution will be 640x480

---

## 4. Install Docker Engine

Install Docker on Ubuntu 24.04 following the official steps:

### Add Docker’s GPG key & repository

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
   https://download.docker.com/linux/ubuntu \
   $(. /etc/os-release && echo \"${UBUNTU_CODENAME}\") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
```

### Install Docker and plugins

```bash
sudo apt-get install -y \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    docker-buildx-plugin \
    docker-compose-plugin
```

---

## 5. Run the Docker Container

Launch the container with GPU and device access:

```bash
# Allow X11 forwarding from Docker:
export DISPLAY=:0
xhost +local:docker

# Run the ROS 2 Jazzy container:
sudo docker run -it --privileged \
    --net=host \
    -v /dev:/dev \
    -v /run/udev:/run/udev \
    --group-add video \
    ghcr.io/nautilus-unipd/jetson-nano-setup:latest
```

Inside, Ubuntu 24.04 and ROS 2 Jazzy are ready.

---

## 6. Connecting to the Jetson Nano

Depending on your network topology, use SSH or VNC as follows:

### 6.1 Jetson connected directly to your PC

* **SSH:**

  ```bash
  ssh user@jetson-nano.local
  ```
* **VNC:**
  Connect your VNC client to `jetson-nano.local:5900`.

### 6.2 Jetson on the local network

Use the same commands as in **6.1**, since mDNS (`.local`) resolves on LAN.

### 6.3 Jetson behind a Raspberry Pi (as a gateway)

If your Jetson is connected to a Raspberry Pi and you need to:

* **SSH via Raspberry Pi jump host:**

  ```bash
  ssh -J admin@raspberrypi.local user@jetson-nano.local
  ```

* **VNC via SSH tunnel:**

  ```bash
  ssh -L 5901:jetson-nano.local:5900 admin@raspberrypi.local
  ```

  Then point your VNC client to `localhost:5901`.

> **Internet access through Raspberry Pi**: To enable the Jetson Nano to reach the Internet via the Raspberry Pi, refer to the networking section in the `raspberry-setup` repo README.
