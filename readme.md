#Share-links.biz Captcha Solver

##Usage:

Currently uses a hardcoded share-links.biz link for testing. Just replace it in the code line 75.
When the script is executed it tries to find out the bit letter digit combination in the background and prints in in white big letters onto the output image.
Then it tries to find the matching small letters and generates a position which needs to be clicked and displays it as a white dot.
All resulting images are saved in the results folder. I did't do any real statisics on the success rate of the script yet. But it seems to work well enough to generally overcome the captchas after 3-4 tries.

##Dependencies


Windows:
Open CV 2.4.10: http://sourceforge.net/projects/opencvlibrary/?source=typ_redirect
				Copy cv2.pyd to C:/Python27/lib/site-packages.
				
Numpy 1.9.1: http://sourceforge.net/projects/numpy/files/NumPy/1.9.1/numpy-1.9.1-win32-superpack-python2.7.exe/download

easy_install Pillow

For ocr:

	matplotlib 1.4.3: https://downloads.sourceforge.net/project/matplotlib/matplotlib/matplotlib-1.4.3/windows/matplotlib-1.4.3.win32-py2.7.exe
	
	scipy: 0.15.1: http://sourceforge.net/projects/scipy/files/scipy/0.15.1/
	
	pip install scikit-learn python-dateutil pytz pyparsing six --force-reinstall --upgrade
	
	
Ubuntu/Linux:
Not tried but should work.
Install everything with pip

Author: FiveF