## Client
The client can be launched from any system (PC, laptop, etc.) that is connected to the network where the Jetson is located. It allows you to view the processing performed by the server and the distance data.
To launch the script, run: `python -m client.webClientArUcoMarker`

## Configuring Client Settings
It may be necessary to modify `DEFAULT_SERVER_IP`, which is the server IP address in the `ClientConfig` class. `DEFAULT_SERVER_IP` must match the IP address of the machine (Jetson) where webServerArUcoMarker.py is running.
