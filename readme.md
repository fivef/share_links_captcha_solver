#Share-links.biz Captcha Solver

Author: FiveF

##Usage:

`captcha.py [path to image of any type] [show]`

`captcha.py` - downloads a predefined container and tries to solve the captcha. Results are shown in a result window.

`captcha.py <image_path_to_captcha_image>` - solves this image and prints the x and y coordinates of the point to click after XY_RESULTS

`captcha.py <image_path_to_captcha_image> show` - the show parameter shows a window with the resulting image for debugging

##Description

When the script is executed it tries to find out the big letter-digit combination in the background and prints it in white big letters onto the output image.
Then it tries to find the matching small letters and generates a position which needs to be clicked and displays it as a white dot. The x and y values of the point to click are printed after XY_RESULT.
All resulting images are saved into the results folder. 
I haven't done any real statisics on the success rate of the script yet. But it seems to work far above 50%.

##Dependencies

###Windows:	
`pip install -r pip_requirements.txt`


####If install with pip fails:

- Open CV 2.4.10: http://sourceforge.net/projects/opencvlibrary/?source=typ_redirect
				Copy cv2.pyd to C:/Python27/lib/site-packages.

- Numpy 1.9.1: http://sourceforge.net/projects/numpy/files/NumPy/1.9.1/numpy-1.9.1-win32-superpack-python2.7.exe/download

- easy_install Pillow

###Ubuntu/Linux:
Not tried but should work.
`pip install -r pip_requirements.txt`

