All the image processing code is inside main.py. To run, there are a few options:

	python main.py [-v] [-d] [-i INPUT FILES] [-a ACTION] [-x XLENGTH] [-y YLENGTH]

'-a', '--action'   - 'calibrate' or 'rectify' or 'rectify-square'
	'calibrate' - moves images taken from Arduino and saved by the server code to a new folder cal/, also saves a text file containing the dimensions of the box used for calibration
	'poll' - continuously wait for new images and finds the dimensions of box
	'find_dim' - finds the dimension of a box
'-i', '--input'    - input files to use
'-y', '--ylen'    - y length for calibration
'-x', '--xlen'    - x length for calibration
'-d', '--disp'     - prints images if flag turned on
'-v', '--verbose'  - prints output if flag turned on

To use "poll", run this code side by side with the server script.

The server.py file contains the code that continuously waits and saves images send by the Arduino. Start this script if you want to save images from the Arduino onto your computer.