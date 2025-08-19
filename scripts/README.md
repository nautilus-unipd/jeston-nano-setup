## How to use calibration.py 
The `calibration.py` generates the `stereo_calibration_data.pkl` file. This script must be run from a PC connected to a BlackBox (Raspberry Pi),
as the calibration is performed by directly accessing the stereo stream from the Raspberry.

> **NOTE:** Calibration should be repeated periodically, and also whenever the image quality changes (e.g., from 480p to 1080p).

To use the script, run: ```python3 calibration.py --help``` The two main functions are:
- `python3 calibration.py --calibrate` Starts the calibration process and saves the data to the .pkl file.
- `python3 calibration.py --upload` Uploads the .pkl file directly to the Jetson using SCP.

## Configuring Calibration Settings
In the `Config` class, it may be necessary to modify the following parameters:

- `STREAM_URL`: Set the IP address of the Raspberry Pi to access the raw video stream without processing.
- `SQUARE_SIZE`: It is very important to provide an accurate value representing the size (in cm) of each square on the chessboard.
- `DEFAULT_SERVER_IP`: This is the IP address of the Jetson device, used to upload the .pkl file.
- `JETSON_PATH`: The path on the Jetson where the file will be saved.

It is also possible to modify `MIN_CALIBRATION_IMAGES` to specify the desired number of images to be acquired for calibration
