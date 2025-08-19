## Server Folder
This folder is mounted into the ROS2 container by the `run_container.sh` script, so all files inside it will be visible from within the container.

## How to run
After entering the container using the ```./run_container.sh``` in the shell, you can start the server by running the Python script ```python3 webServerArUcoMarker.py```. Once the server is running, the following endpoints are available:
- `/video_feed` (GET request): provides the video stream with correction already applied (based on calibration parameters)
- `/detection_data` (GET request): retrieves the detection data
- `/status` (GET request): displays the current status of the service
- `/reset_filters` (POST request): resets the smoothing filter
- `/shutdown` (POST request): shuts down the service

## Server Parameter Configuration
In the `Config` class may be necessary to modify `STREAM_URL`, which represents the address of the Raspberry Pi from which the frames are retrieved (in this case, simply replace 10.70.64.50 with the correct address on the local network), 
and `CALIBRATION_FILE` if the calibration file has been changed.
