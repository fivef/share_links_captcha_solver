import cv2
import urllib2
import numpy as np
import math
import datetime
import os
import collections
import random

from PIL import Image

debug = True

show_images = False

#enable full array prints
#np.set_printoptions(threshold=None)

folder_name = 'Bilder'
folder_results = 'results'
labeled_letters_folder = 'labeled_letters'
labeled_numbers_folder = 'labeled_numbers'
labeled_little_numbers = 'labeled_little_numbers'
little_numbers_archive = 'little_numbers_archive'
temp_little_numbers = 'temp_little_numbers'

timestamp = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")


def read_backgroud_letters(captcha):
	captcha_backgroud_letters = extract_backgroud_letters(captcha)
	
	letter, number = make_letters_horizontal_and_split_into_two(captcha_backgroud_letters)
	
	letter = crop_to_bounding_rect(letter)
	number = crop_to_bounding_rect(number)
		
	cv2.imwrite(os.path.join(folder_name, timestamp + '_letter.png'), letter)
	cv2.imwrite(os.path.join(folder_name, timestamp + '_number.png'), number) 
		
	#im = Image.open(os.path.join(folder_name, timestamp + '_number.png'))
	#im = Image.open(os.path.join(folder_name, timestamp + '_letter.png'))
	
	letter_string = get_letter_in_image_by_comparision(letter)
	number_string = get_number_in_image_by_comparision(number)
	
	#output text
	print "Output text: ",letter_string + ' ' + number_string
	return letter_string, number_string

def download_image():
	
	'''
	headers = {
		'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:2.0.1) Gecko/2010010' \
		'1 Firefox/4.0.1',
		'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
		'Accept-Language':'en-us,en;q=0.5',
		'Accept-Charset':'ISO-8859-1,utf-8;q=0.7,*;q=0.7'}
	'''	
	
	headers = {'Host': 'share-links.biz',
		'Connection': 'keep-alive',
		'Cache-Control': 'max-age=0',
		'Accept': 'image/webp,*/*;q=0.8',
		'User-Agent': 'Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/29.0.1547.62 Safari/537.36',
		'Referer': 'http://share-links.biz/_h379bkmu0zd',
		'Accept-Encoding': 'gzip,deflate,sdch',
		'Accept-Language': 'de-DE,de;q=0.8,en-US;q=0.6,en;q=0.4',
		'Cookie': '__utma=251446340.577644118.1353529793.1353529793.1353529793.1; PHPSESSID=2j7t3f0tk1hvus5fr4bsslav02; lastVisit=1378390568; SLlng=de'}


	

	req = urllib2.Request('http://share-links.biz/_mt5ac0nm5tb', None,
							headers)
	f = urllib2.urlopen(req)
	page = f.read()

	#tree = lxml.html.fromstring(page)
	
	#imgurl = "http://www.amaderforum.com/" + \
	#		tree.xpath(".//img[@id='imagereg']")[0].get('src')
			
	imgurl = "http://share-links.biz/captcha.gif?d=1378390568&PHPSESSID=2j7t3f0tk1hvus5fr4bsslav02"

	req = urllib2.Request(imgurl, None, headers)
	f = urllib2.urlopen(req)
	img = f.read()

	open(os.path.join(folder_name, timestamp + '_downloaded.gif'), 'wb').write(img)
	
def convert_gif_to_png():
	im = Image.open(os.path.join(folder_name, timestamp + '_downloaded.gif'))
	#im.convert('RGB')

	print im.save(os.path.join(folder_name, timestamp + '_downloaded.png'), 'PNG')

def extract_backgroud_letters(captcha):
		
	captcha = cv2.cvtColor(captcha, cv2.COLOR_BGR2RGB)
	
	colors_to_remove = []
	colors_to_remove.append(np.array([100,94,76]))

	mask = None
	
	print captcha
	
	#creat a mask of all removed colors
	for color in colors_to_remove:
		
		new_mask = cv2.inRange(captcha,color , color)
		
		if mask is None:
			mask = new_mask
			
		mask = cv2.add(mask, new_mask)

	#invert mask
	#mask = cv2.bitwise_not(mask)
	
	#filter out the masked pixels
	mask = cv2.bitwise_and(captcha,captcha,mask = mask)

	mask = cv2.cvtColor(mask, cv2.cv.CV_RGB2GRAY)
	
	threshold, mask = cv2.threshold(mask, 50, 255 , cv2.THRESH_BINARY)

	#erode 
	element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
	eroded = cv2.erode(mask, element)
	
	#todo 4,4 seems to be better because with 10,10 the contours are lost
	element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
	dilated = cv2.dilate(eroded,element)
	
	#get rid of the top pixel stripe
	dilated = dilated[10:,:]
	
	cv2.imwrite(os.path.join(folder_name, timestamp + '_dilated.png'), dilated)

	
	return dilated

def make_letters_horizontal_and_split_into_two(captcha_backgroud_letters):

	#Get the two cluster centers
	centers = calc_2_means(captcha_backgroud_letters)
	
	#Make letter horizontal
	if(centers[0][0] > centers[1][0]):
		point = centers[0] - centers[1]
	else:
		point = centers[1] - centers[0]
		
	angle = math.degrees(math.atan2(point[1], point[0]))
	print "rotation angle: ", angle
	dilated_output = captcha_backgroud_letters.copy()

	#draw a white circle with 4 radius and fill it (-1)
	cv2.circle(dilated_output,(int(centers[0][0]),int(centers[0][1])),5, (255,255, 255),-1)
	cv2.circle(dilated_output,(int(centers[0][0]),int(centers[0][1])),3, (0,0, 0),-1)
	
	cv2.circle(dilated_output,(int(centers[1][0]),int(centers[1][1])),5, (255,255, 255),-1)
	cv2.circle(dilated_output,(int(centers[1][0]),int(centers[1][1])),3, (0,0, 0),-1)
	
	#cv2.imshow("dilated_output", dilated_output)
	#cv2.waitKey()
	
	#get center of rotation (center between both clusters)
	rot_center = centers[0]/2 + centers[1]/2
	
	#rotate
	image = captcha_backgroud_letters
	rot_mat = cv2.getRotationMatrix2D((rot_center[0],rot_center[1]),angle,1.0)
	rotated = cv2.warpAffine(image, rot_mat, (len(image[0]),len(image)),flags=cv2.INTER_LINEAR)
	
	#split letter and number
	letter = rotated[:,:rot_center[0]]
	number = rotated[:,rot_center[0]:]
	
	return letter, number

def calc_2_means(captcha_backgroud_letters):
	
	white_points_image = get_non_zero_points_from_image(captcha_backgroud_letters)
	
	# define criteria and apply kmeans()

	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

	#cv2.kmeans(data, K, criteria, attempts, flags[, bestLabels[, centers]])	retval, bestLabels, centers

	best_labels = np.array([1,2,3])
	
	number_of_clusters = 2
	
	attempts = 10
	
	initial_centers_flags = cv2.KMEANS_RANDOM_CENTERS

	ret,best_labels,centers = cv2.kmeans(white_points_image,number_of_clusters, criteria, attempts ,initial_centers_flags)

	print "centers", centers

	return centers

def crop_to_bounding_rect(image):

	white_points_image = get_non_zero_points_from_image(image)
	
	#crop
	x,y,w,h = cv2.boundingRect(white_points_image)
	return image[y:y+h,x:x+w]
	
def get_non_zero_points_from_image(image):
	(y_non_zero_array,x_non_zero_array) = np.nonzero(image > 0)

	#stack both arrays vertically
	Z = np.vstack((x_non_zero_array,y_non_zero_array))

	#transpose to fit into kmeans (expects features in columns)
	Z = Z.T

	#convert to np.floeat32
	Z = np.float32(Z)

	#bring array in correct shape for boundingRect (3 dimensionsl	[[[x,y]] [[x,y]]] )
	Z = Z.reshape(len(Z),1,2)

	return Z

def get_letter_in_image_by_comparision(image):
	return get_text_in_image_by_comparision(image, labeled_letters_folder)
	
def get_number_in_image_by_comparision(image):
	return get_text_in_image_by_comparision(image, labeled_numbers_folder)

def get_little_number_label_by_comparison(image):
	return get_text_in_image_by_comparision(image, temp_little_numbers)
	
#reads the text in "image" by comparing image to the images in the "labeled_images_folder_param"	
def get_text_in_image_by_comparision(image, labeled_images_folder_param):
	
	print type(image)
	print image
	
	if image is None:
		raise Exception('Input search image is None')
	
	labeled_images_folder = labeled_images_folder_param
	
	#compare image to all images in the labeled_images folder and return the name of the best match

	image = resize_to_square_image(image)

	if show_images:
		cv2.imshow("image", image)

	#compare letter to all images
	dirs = os.listdir(labeled_images_folder)
	
	results = {}
	
	for filename in dirs:

		compare_image = cv2.imread(os.path.join(labeled_images_folder ,filename), cv2.CV_LOAD_IMAGE_GRAYSCALE)
		if(compare_image is None):
			raise Exception('Compare image is None')
		
		compare_image = resize_to_square_image(compare_image)
	 
		compare_matrix = cv2.absdiff(compare_image, image)

		results[filename] = cv2.sumElems(compare_matrix)
		
	results = collections.OrderedDict(sorted(results.items(), key=lambda t: t[1], reverse=False))

	for filename, distance in results.items():
		print filename, distance

	best_match = results.keys()[0]
	
	if(debug):
		best_match_image = cv2.imread(os.path.join(labeled_images_folder,best_match), cv2.CV_LOAD_IMAGE_GRAYSCALE)
		
		best_match_image = resize_to_square_image(best_match_image)
		if show_images:
			cv2.imshow("best_match_image", best_match_image)
	
	
	#remove the filename extension
	split_best_match = best_match.split('.')
	
	best_match = split_best_match[0]
	
	print "Best match:",best_match
	return best_match


#make a square binary image out of a portrait or landscape binary image
#img: 2 D numpy array (binary image)
#desired_size: square side length default 150
#image_threshold: binary threshold default 160
#returns: 2D numpy array (binary image) square with side length = desired_size
def resize_to_square_image(img, desired_size = 100, image_threshold = 140):
	 
	height, width = np.shape(img)
	
	if height > width:
		print 'portrait image'
		
		hpercent = (desired_size / float(height))
		wsize = int((float(width) * float(hpercent)))
		img = cv2.resize(img,(wsize, desired_size), interpolation=cv2.INTER_NEAREST)

		new_height, new_width = np.shape(img)
		
		#create black padded_image
		padded_image = np.zeros((desired_size,desired_size), np.uint8)
		
		start_column = (desired_size - new_width)/2
		padded_image[:,start_column:start_column + new_width] = img
		
	else:
		print 'landscape image'
		
		hpercent = (desired_size / float(width))
		wsize = int((float(height) * float(hpercent)))
		img = cv2.resize(img,(desired_size, wsize), interpolation=cv2.INTER_NEAREST)

		new_height, new_width = np.shape(img)
		
		#create black padded_image
		padded_image = np.zeros((desired_size,desired_size), np.uint8)
		
		start_column = (desired_size - new_height)/2
		padded_image[start_column:start_column + new_height,:] = img

	threshold, padded_image = cv2.threshold(padded_image, image_threshold, 255 , cv2.THRESH_BINARY)

	return padded_image
	
	
def extract_foreground_letters(captcha):

	captcha = cv2.cvtColor(captcha, cv2.COLOR_BGR2RGB)
	
	colors_to_remove = []
	
	colors_to_remove.append(np.array([140,126,100]))
	colors_to_remove.append(np.array([140,130,100]))
	colors_to_remove.append(np.array([100,94,76]))
	colors_to_remove.append(np.array([100,94,68]))
	colors_to_remove.append(np.array([68,98,108]))
	colors_to_remove.append(np.array([100,126,140]))
	colors_to_remove.append(np.array([108,98,76])) #last
	colors_to_remove.append(np.array([108,98,74]))
	colors_to_remove.append(np.array([132,118,92]))
	colors_to_remove.append(np.array([108,134,148]))
	colors_to_remove.append(np.array([80,110,124]))
	colors_to_remove.append(np.array([92,118,124]))
	colors_to_remove.append(np.array([124,118,92]))
	colors_to_remove.append(np.array([94,114,124]))
	colors_to_remove.append(np.array([100,122,132]))
	colors_to_remove.append(np.array([76,102,116]))
	
	mask = None
	
	#creat a mask of all removed colors
	for color in colors_to_remove:
		
		new_mask = cv2.inRange(captcha,color , color)
		
		if mask is None:
			mask = new_mask
			
		mask = cv2.add(mask, new_mask)

	#invert mask
	mask = cv2.bitwise_not(mask)
	
	#filter out the masked pixels
	subtracted = cv2.bitwise_and(captcha,captcha,mask = mask)

	#erode 
	element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
	eroded = cv2.erode(mask, element)
	
	extracted_foreground = eroded

	return extracted_foreground
	
#returns position of capital letter in alphabet starting with A = 0
def get_letter_pos_in_alphabet(letter):
	
	number_in_alphabet = ord(letter) - 65
	
	if number_in_alphabet < 0 or number_in_alphabet > 26:
		raise Exception('The letter ' + letter + 'is not in capitals letters alphabet')
	print 'letter ordinality', number_in_alphabet
	return number_in_alphabet

#detect cut indexes:
def get_row_and_column_cut_indexes(extracted_foreground_letters):
	extracted_foreground_letters_with_lines = np.array(extracted_foreground_letters)
	
	#split rows:
	
	row_cut_indexes = [0] #add 0 as first cut
	
	old_index = 0
	
	old_cut_index = 0
	
	free_space_counter = 0
	
	for index, row in enumerate(extracted_foreground_letters):
		non_zero_count = cv2.countNonZero(row)
		if(non_zero_count < 8):
	
			if((index - old_index) <= 2):
			 
				#increment free space counter
				free_space_counter += 1
			else:
				
				if free_space_counter > 3:
					cut_index = old_index - int((free_space_counter / 2))
	
					if((cut_index - old_cut_index) >= 10):
						row_cut_indexes.append(cut_index)
					
						cv2.line(extracted_foreground_letters_with_lines,(0,cut_index),(len(extracted_foreground_letters_with_lines[0]),cut_index), (255,255,255))
					
					old_cut_index = cut_index
					
					free_space_counter = 0
				
			old_index = index
			
	row_cut_indexes.append(len(extracted_foreground_letters)) #add bottom end
			
	#split columns:

	min_free_space_needed_to_cut_threshold = 7 
	column_cut_indexes = [0] # add 0 column as leftmost value
	old_index = 0
	old_cut_index = 0
	free_space_counter = 0
	index = 0
	while index < len(extracted_foreground_letters[0]):
		
		non_zero_count = cv2.countNonZero(extracted_foreground_letters[:,index])
		
		#TODO adapt this value if no lines are drawn by row cut
		if(non_zero_count < 2):
	
			if((index - old_index) <= 2):
				
				#increment free space counter
				free_space_counter += 1
			else:
				
				if free_space_counter > 3:
					cut_index = old_index - int((free_space_counter / 2))
					
					if((cut_index - old_cut_index) >= min_free_space_needed_to_cut_threshold):
						column_cut_indexes.append(cut_index)
					
						cv2.line(extracted_foreground_letters_with_lines,(cut_index,0),(cut_index, len(extracted_foreground_letters_with_lines)), (255,255,255))
	
					old_cut_index = cut_index
					free_space_counter = 0
				
			old_index = index
		index += 1
	
	column_cut_indexes.append(len(extracted_foreground_letters[0])) #add last column 
	
	cv2.imwrite(os.path.join(folder_name, timestamp + '_extracted_foreground_letters_with_lines.png'), extracted_foreground_letters_with_lines)
	
	return row_cut_indexes, column_cut_indexes

#===============================================================================
# the following functions are not used anymore
#===============================================================================
	
def find_contours_and_save_as_seperate_images(image):
	im = image
	contours, hierarchy = cv2.findContours(im,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_TC89_L1)

	for i in range(0, len(contours)):
		if (i % 2 == 0):
			cnt = contours[i]
			#mask = np.zeros(im2.shape,np.uint8)
			#cv2.drawContours(mask,[cnt],0,255,-1)
			x,y,w,h = cv2.boundingRect(cnt)
			#cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
			letter = im[y:y+h,x:x+w]
			#cv2.imshow('Features', im)
			#cv2.waitKey()
			
			cv2.imwrite(str(i)+'.png', letter)


'''
input: [r,g,b]
output: prints hsv
'''
def calc_hsv_from_rgb(rgb_array):
	color = np.uint8([[rgb_array]])
	hsv = cv2.cvtColor(color,cv2.COLOR_RGB2HSV)
	print hsv
	
	
def convert_numpy_image_to_cv_mat_32FC1(numpy_image):
	#convert numpy image to cv mat CV_32FC1 image
	numpy_image = cv2.cvtColor(numpy_image, cv2.cv.CV_BGR2GRAY)
	numpy_image = numpy_image.astype(np.float32)
	
	dest = cv2.cv.CreateMat(len(numpy_image), len(numpy_image[0]), cv2.cv.CV_32FC1)
	src = cv2.cv.fromarray(numpy_image)
	cv2.cv.Convert(src, dest)
	return dest



		
def main():
	download_image()
	
	convert_gif_to_png()
	
	captcha = cv2.imread(os.path.join(folder_name, timestamp + '_downloaded.png'))
		
	letter_string, number_string = read_backgroud_letters(captcha)
	
	extracted_foreground_letters = extract_foreground_letters(captcha)
	
	#split foreground letters into single Letter Number Parts:
	
	row_cut_indexes, column_cut_indexes = get_row_and_column_cut_indexes(extracted_foreground_letters)
	
	print "rows cut indesxes", row_cut_indexes
	print "column cut indexes", column_cut_indexes
	
	#cut out desired column
	letter_ordinality = get_letter_pos_in_alphabet(letter_string)
	
	
	print 'row cut indexes', row_cut_indexes[letter_ordinality]
	upper_row_cut_index = row_cut_indexes[letter_ordinality]
	lower_row_cut_index = row_cut_indexes[letter_ordinality+1]
	desired_column_image = extracted_foreground_letters[upper_row_cut_index:lower_row_cut_index,:]
	
	desired_column_image_full = captcha[upper_row_cut_index:lower_row_cut_index,:]
	
	
	
	#cut out letter number pairs
	
	letter_number_images = []
	
	print "column_cut_index_length", len(column_cut_indexes)
	print column_cut_indexes
	
	for index in range(len(column_cut_indexes)-1):
		letter_number_images.append(desired_column_image[:,column_cut_indexes[index]:column_cut_indexes[index+1]])
		
	 
		
	
	print "Number of images ", len(letter_number_images)
	
	number_images = []
	
	#delete all files in temp_little_numbers dir
	
	files = os.listdir(temp_little_numbers)
	
	for filename in files:
		os.remove(os.path.join(temp_little_numbers,filename))
	
	for index, image in enumerate(letter_number_images):
		letter_image, number_image = make_letters_horizontal_and_split_into_two(image)
		 
		number_image = crop_to_bounding_rect(number_image)
		
		number_images.append(number_image)
			
		cv2.imwrite(os.path.join(temp_little_numbers,str(index) + '.png'), number_image)
		
		cv2.imwrite(os.path.join(little_numbers_archive, timestamp + str(index) + '.png'), number_image)
		 
		#cv2.imshow("number", number)
		#cv2.waitKey()
		
	#load image of searched number
	
	print "path to searched number image", str(os.path.join(labeled_little_numbers,str(number_string))) + '.png'
	
	searched_number = cv2.imread(os.path.join(labeled_little_numbers,str(number_string) + '.png'),0)
	
	if searched_number is None:
		raise Exception('Image ' + labeled_little_numbers + str(number_string) + '.png' + ' could not be found. Is there a labled_little_numbers template for number ' + str(number_string) + '?')
	
	number = int(get_little_number_label_by_comparison(searched_number))
	
	
	print "number", number
	
	
	#determine the point to click
	#randomize it to prevent detection
	
	tile_margin = 10
	
	left_column_cut_index = column_cut_indexes[number]
	right_column_cut_index = column_cut_indexes[number+1]
	
	x_position = random.randint(left_column_cut_index + tile_margin,right_column_cut_index - tile_margin)
	
	y_position = random.randint(upper_row_cut_index + tile_margin, lower_row_cut_index - tile_margin)
	
	print x_position
	
	
	print y_position
	
	#create output image with click circle and recognized number + letter
	cv2.circle(captcha,(x_position,y_position),5,(255,255,255),-1)
	cv2.putText(captcha, letter_string + ' ' + number_string , (len(captcha[0])/2,len(captcha)/2), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255))
	
	cv2.imwrite(os.path.join(folder_results, timestamp + '_result.png'), captcha)
	

	cv2.imshow("captcha", captcha)
	
	cv2.waitKey()
	

if __name__ == "__main__":
	
	main()
		









