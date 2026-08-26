This package reads a photogate signal from a USB serial device.

Find the serial device: `ls /dev/ttyUSB*`

Grant access: `sudo chmod 666 /dev/ttyUSB0`

Run: `roslaunch photogate_reader photo_gate.launch`

Topic: `/photogate_reader/photogate_state`
