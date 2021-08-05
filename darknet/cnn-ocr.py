#!/usr/bin/env python
# -*- encoding: iso-8859-1 -*-

# Adapted by Rayson Laroca

#!python3
'''
Python 3 wrapper for identifying objects in images

Requires DLL compilation

Both the GPU and no-GPU version should be compiled; the no-GPU version should be renamed "yolo_cpp_dll_nogpu.dll".

On a GPU system, you can force CPU evaluation by any of:

- Set global variable DARKNET_FORCE_CPU to True
- Set environment variable CUDA_VISIBLE_DEVICES to -1
- Set environment variable "DARKNET_FORCE_CPUU" to "true"

To use, either run perform_detection() after import, or modify the end of this file.
See the docstring of perform_detection() for parameters.

Directly viewing or returning bounding-boxed images requires OpenCV to be installed 

Original *nix 2.7: https://github.com/pjreddie/darknet/blob/0f110834f4e18b30d5f101bf8f1724c34b7b83db/python/darknet.py
Windows Python 2.7 version: https://github.com/AlexeyAB/darknet/blob/fc496d52bf22a0bb257300d3c79be9cd80e722cb/build/darknet/x64/darknet.py
'''

from __future__ import print_function
from ctypes import *

import os
import cv2
import math
import time
import argparse
import operator
import numpy as np

if not os.path.exists('./darknet.so'):
	print('Darknet is not compiled; compile it using \'make\'!')
	exit(1)

def get_args():
	ap = argparse.ArgumentParser()
	ap.add_argument('-p', '--path', required = False, default = './input-images/', help = 'path of the input images [default = \'./input-images/\']')
	ap.add_argument('-c', '--cfg', required = False, default = './cfg/cr-net.cfg', help = 'cfg file [default = \'./cfg/cr-net.cfg\']')
	ap.add_argument('-w', '--weights', required = False, default = './weights/cr-net.weights', help = 'weights file [default = \'./weights/cr-net.weights\']')
	ap.add_argument('-d', '--data', required = False, default = './data/obj.data', help = 'data file [default = \'./data/obj.data\']')
	ap.add_argument('-bb', '--black_borders', required = False, type = str2bool, default = True, help = 'add black borders to the license plate patches so that they have an aspect ratio (w/h) between 2.5 and 3.0. In this way, the network processes less distorted images (the aspect ratio of the input image is 2.75) [default = True]')
	ap.add_argument('-r', '--results', required = False, default = './output.txt', help = 'path of the results file [default = \'./output.txt\']')
	ap.add_argument('-s', '--show_img', required = False, type = str2bool, default = False, help = 'compute (and show) the bounding boxes [default = False]')
	args = vars(ap.parse_args())

	return args['path'], args['cfg'], args['weights'], args['data'], args['black_borders'], args['results'], args['show_img']

def str2bool(value):
	if isinstance(value, bool):
		return value
	if value.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif value.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

class BOX(Structure):
	_fields_ = [("x", c_float),
				("y", c_float),
				("w", c_float),
				("h", c_float)]

class DETECTION(Structure):
	_fields_ = [("bbox", BOX),
				("classes", c_int),
				("prob", POINTER(c_float)),
				("mask", POINTER(c_float)),
				("objectness", c_float),
				("sort_class", c_int)]

class IMAGE(Structure):
	_fields_ = [("w", c_int),
				("h", c_int),
				("c", c_int),
				("data", POINTER(c_float))]

class METADATA(Structure):
	_fields_ = [("classes", c_int),
				("names", POINTER(c_char_p))]

hasGPU = True
if os.name == "nt":
	cwd = os.path.dirname(__file__)
	os.environ['PATH'] = cwd + ';' + os.environ['PATH']
	winGPUdll = os.path.join(cwd, "yolo_cpp_dll.dll")
	winNoGPUdll = os.path.join(cwd, "yolo_cpp_dll_nogpu.dll")
	envKeys = list()
	for k, v in os.environ.items():
		envKeys.append(k)
	try:
		try:
			tmp = os.environ["FORCE_CPU"].lower()
			if tmp in ["1", "true", "yes", "on"]:
				raise ValueError("ForceCPU")
			else:
				print("Flag value '" +tmp+ "' not forcing CPU mode")
		except KeyError:
			# we never set the flag
			if 'CUDA_VISIBLE_DEVICES' in envKeys:
				if int(os.environ['CUDA_VISIBLE_DEVICES']) < 0:
					raise ValueError("ForceCPU")
			try:
				global DARKNET_FORCE_CPU
				if DARKNET_FORCE_CPU:
					raise ValueError("ForceCPU")
			except NameError:
				pass
		if not os.path.exists(winGPUdll):
			raise ValueError("NoDLL")
		lib = CDLL(winGPUdll, RTLD_GLOBAL)
	except (KeyError, ValueError):
		hasGPU = False
		if os.path.exists(winNoGPUdll):
			lib = CDLL(winNoGPUdll, RTLD_GLOBAL)
			print("Notice: CPU-only mode")
		else:
			# try the other way, in case no_gpu was compile but not renamed
			lib = CDLL(winGPUdll, RTLD_GLOBAL)
			print("Environment variables indicated a CPU run, but we didn't find '"+winNoGPUdll+"'. Trying a GPU run anyway.")
else:
	lib = CDLL("./darknet.so", RTLD_GLOBAL)

lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

if hasGPU:
	set_gpu = lib.cuda_set_device
	set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

net_main = None
meta_main = None
altNames = None

ndarray_image = lib.ndarray_to_image
ndarray_image.argtypes = [POINTER(c_ubyte), POINTER(c_long), POINTER(c_long)]
ndarray_image.restype = IMAGE

def nparray_to_image(img):
    data = img.ctypes.data_as(POINTER(c_ubyte))
    image = ndarray_image(data, img.ctypes.shape, img.ctypes.strides)

    return image

def add_black_borders(img, min_ratio=2.5, max_ratio=3.0):
	h, w = np.shape(img)[:2]
	ar = float(w)/h

	bw = 0
	bh = 0

	if ar < min_ratio:
		while ar < min_ratio:
			bw += 1
			ar = float(w+bw)/(h+bh)
	else:
		while ar > max_ratio:
			bh += 1
			ar = float(w)/(h+bh)

	return cv2.copyMakeBorder(img, bh//2, bh//2, bw//2, bw//2, cv2.BORDER_CONSTANT, value=(0, 0, 0))

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=0.25, debug=False, black_borders=True):
	if black_borders:
		img = cv2.imread(image.decode("ascii"))
		img = add_black_borders(img)
		img = nparray_to_image(img)
	else:
		img = load_image(image, 0, 0)
	
	if debug: 
		print("Loaded image")

	num = c_int(0)
	if debug: 
		print("Assigned num")

	pnum = pointer(num)
	if debug: 
		print("Assigned pnum")

	predict_image(net, img)
	if debug: 
		print("did prediction")

	dets = get_network_boxes(net, img.w, img.h, thresh, hier_thresh, None, 0, pnum, 0)
	if debug: 
		print("Got dets")

	num = pnum[0]
	if debug:
		print("got zeroth index of pnum")

	if nms:
		do_nms_sort(dets, num, meta.classes, nms)

	if debug: 
		print("did sort")

	res = []
	if debug: 
		print("about to range")

	for j in range(num):
		if debug: 
			print("Ranging on "+str(j)+" of "+str(num))
			print("Classes: "+str(meta), meta.classes, meta.names)

		for i in range(meta.classes):
			if debug: print("Class-ranging on " + str(i) + " of " + str(meta.classes) + "= " + str(dets[j].prob[i]))
			if dets[j].prob[i] > 0:
				b = dets[j].bbox

				# coordinates are around the center
				b.x = b.x - int(round(b.w/2))
				b.y = b.y - int(round(b.h/2))

				if altNames is None:
					nameTag = meta.names[i]
				else:
					nameTag = altNames[i]

				if debug:
					print("Got bbox", b)
					print(nameTag)
					print(dets[j].prob[i])
					print((b.x, b.y, b.w, b.h))

				res.append((nameTag, dets[j].prob[i], int(round(b.x)), int(round(b.y)), int(round(b.x+b.w)), int(round(b.y+b.h)))) # (label, confidence, x1, y1, x2, y2)

	if debug: 
		print("did range")

	res = sorted(res, key=lambda x: -x[1])
	if debug: 
		print("did sort")

	free_image(img)
	if debug: 
		print("freed image")

	free_detections(dets, num)
	if debug: 
		print("freed detections")

	return res

def perform_detection(img_path='input-images/3327.jpg', thresh=0.5, cfg='cfg/cr-net.cfg', weights='weights/cr-net.weights', data='data/obj.data', black_borders=True):
	"""
	Convenience function to handle the detection and returns of objects.

	Displaying bounding boxes requires libraries cv2 and numpy

	Parameters
	----------------
	img_path: str
		Path to the image to evaluate. Raises ValueError if not found

	thresh: float (default= 0.5)
		The detection threshold

	cfg: str
		Path to the configuration file. Raises ValueError if not found

	weights: str
		Path to the weights file. Raises ValueError if not found

	data: str
		Path to the data file. Raises ValueError if not found

	black_borders: Boolean
		Add black borders to the license plate patch so that it has an aspect ratio (w/h) between 2.5 and 3.0. In this way, the network processes less distorted images (the aspect ratio of the input image is 2.75)
	
	Returns
    ----------------------

    "img": a numpy array representing an image, compatible with scikit-image
    "detections": list of tuples like ('obj_label', confidence, (x1, y1, x2, y2))
	
	"""

	# Import the global variables. This lets us instance Darknet once, then just call perform_detection() again without instancing again
	global meta_main, net_main, altNames, io #pylint: disable=W0603
	assert 0 < thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
	if not os.path.exists(cfg):
		raise ValueError("Invalid config path '"+os.path.abspath(cfg)+"'")
	if not os.path.exists(weights):
		raise ValueError("Invalid weight path '"+os.path.abspath(weights)+"'")
	if not os.path.exists(data):
		raise ValueError("Invalid data file path '"+os.path.abspath(data)+"'")
	if net_main is None:
		net_main = load_net_custom(cfg.encode("ascii"), weights.encode("ascii"), 0, 1)  # batch size = 1
	if meta_main is None:
		meta_main = load_meta(data.encode("ascii"))
		print('\nResults:')
		print('idx file pred char1 char2 char3 char4 char5 char6 char7')
	if altNames is None:
		# In Python 3, the metafile default access craps out on Windows (but not Linux)
		# Read the names file and create a list to feed to detect
		try:
			with open(data) as metaFH:
				metaContents = metaFH.read()
				import re
				match = re.search("names *= *(.*)$", metaContents, re.IGNORECASE | re.MULTILINE)
				if match:
					result = match.group(1)
				else:
					result = None
				try:
					if os.path.exists(result):
						with open(result) as namesFH:
							namesList = namesFH.read().strip().split("\n")
							altNames = [x.strip() for x in namesList]
				except TypeError:
					pass
		except Exception:
			pass
	if not os.path.exists(img_path):
		raise ValueError("Invalid image path '" + os.path.abspath(img_path) + "'")

	detections = detect(net_main, meta_main, img_path.encode("ascii"), thresh=thresh, black_borders=black_borders)
	return cv2.imread(img_path), detections

def sort_two_rows(detections):
	# motorcycle LPs have 3 letters in one row and 4 digits in another.

	detections = sorted(detections, key=operator.itemgetter(3), reverse=False) # sorts the characters by the vertical position
	
	first_row = detections[:3]
	first_row = sorted(first_row, key=operator.itemgetter(2), reverse=False) # sorts the characters by the horizontal position

	second_row = detections[3:]
	second_row = sorted(second_row, key=operator.itemgetter(2), reverse=False) # sorts the characters by the horizontal position
	
	return first_row + second_row

def detect_double_line(detections):
	# we consider that the characters are arranged in two rows in cases where the bounding boxes of half or more of the predicted characters are located below another character (following https://arxiv.org/abs/1909.01754) or if the aspect ratio (w/h) of the characters is less than 1.5

	if np.shape(detections)[0] == 0: # no characters
		return False

	bb_x1 = []
	bb_y1 = []
	bb_x2 = []
	bb_y2 = []

	y = []

	count = 0
	num_detections = np.shape(detections)[0]

	for d in detections:
		bb_x1.append(d[2])
		bb_y1.append(d[3])
		bb_x2.append(d[4])
		bb_y2.append(d[5])
		y.append((d[3]+d[5])/2) # vertical center of the character

	cy = min(y)
	for d in detections:
		y1 = d[3]
		if y1 >= cy:
			count += 1

	bb_detections = (min(bb_x1), min(bb_y1), max(bb_x2), max(bb_y2))
	aspect_ratio = float(bb_detections[2]-bb_detections[0])/(bb_detections[3]-bb_detections[1])

	if aspect_ratio > 2: 
		return False # we consider license plates with an aspect ratio greater than 2 to have only one row of characters
	elif count >= math.floor(float(num_detections)/2) or aspect_ratio < 1.5:
		return True

	return False

def get_img_paths(path):
	# returns all images in a directory and subdirectory

	if not os.path.exists(path):
		print('Input directory \'{0}\' does not exist!'.format(path))
		exit(1)

	img_paths = []

	for subdir, dirs, files in os.walk(path):
		for file in files:
			file = os.path.join(subdir, file)

			if '.png' in file.lower() or '.jpg' in file.lower() or '.jpeg' in file.lower() or '.tif' in file.lower() or '.tiff' in file.lower():
				img_paths.append(file)

	if np.shape(img_paths)[0] == 0:
		print('No images were found in the input directory (\'{0}\')'.format(path))
		exit(1)

	img_paths.sort()
	img_path_len = max([len(img_path) for img_path in img_paths]) + 1 # the paths of the input images are listed using strings of the same length
	
	return img_paths, img_path_len

def apply_heuristics_for_brazilian_lps(detections):
	detections = detections[:7:] # Brazilian license plates have seven characters (three letters followed by four digits)
	
	if not detect_double_line(detections): # one row
		detections = sorted(detections, key=operator.itemgetter(2), reverse=False) # sorts the characters by the horizontal position
	else: # two rows
		detections = sort_two_rows(detections)

	prediction = ''.join([detection[0] for detection in detections])

	# some swaps of digits and letters, that are often misidentified, are used to improve the recognition
	letters = prediction[:3].replace('0', 'O').replace('1','I').replace('2','Z').replace('4','A').replace('5','S').replace('6','G').replace('7','Z').replace('8','B')
	digits = prediction[3:].replace('A','4').replace('B','8').replace('D','0').replace('G','6').replace('I','1').replace('J','1').replace('O', '0').replace('Q','0').replace('S','5').replace('T','1').replace('Z','7')

	return detections, letters + digits

# code adapted from https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def bb_intersection_over_union(boxA, boxB):
	x1 = max(boxA[0], boxB[0])
	y1 = max(boxA[1], boxB[1])
	x2 = min(boxA[2], boxB[2])
	y2 = min(boxA[3], boxB[3])
 	
	if y2 < y1 or x2 < x1:
		return 0.0
 	
	intersection = max(0, x2-x1) * max(0, y2-y1)

	area1 = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
	area2 = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
 
	return intersection/float(area1+area2-intersection)

# Malisiewicz et al. (code adapted from https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/)
def non_max_suppression_fast(detections, thresh=0.25):
	bboxes = []
	confidences = []

	for label, confidence, x1, y1, x2, y2 in detections:
		w = x2-x1
		h = y2-y1

		if float(w)/h <= 1.5: # geometric constraint
			bboxes.append((label, confidence, x1, y1, x2, y2))

	# if there are no bboxes, return an empty list
	if len(bboxes) == 0:
		return []

	bboxes = sorted(bboxes, key=operator.itemgetter(1), reverse=True) # sort by confidence

	pick = []
	while len(bboxes) > 0:
		pick.append(bboxes[0])
		bboxes.pop(0)

		if bboxes:
			remove = []
			for x in range(np.shape(bboxes)[0]):
				ref = pick[-1:][0][2:6]

				iou = bb_intersection_over_union(ref, bboxes[x][2:6])

				if iou > thresh:
					remove.append(bboxes[x])

			for r in remove:
				bboxes.remove(r)
			remove = []

	return pick

def main():
	path, cfg, weights, data, black_borders, results_file, show_img = get_args()

	img_paths, img_path_len = get_img_paths(path)

	results = ['idx file pred char1 char2 char3 char4 char5 char6 char7\n']
	for idx, img_path in enumerate(img_paths):
		img, detections = perform_detection(img_path=img_path, thresh=0.01, cfg=cfg, weights=weights, data=data, black_borders=black_borders)

		detections = non_max_suppression_fast(detections) # additional non-maximum suppression | darknet's NMS eliminates only objects belonging to the same class, however, objects of different classes may belong to the same class after applying heuristics
		detections, prediction = apply_heuristics_for_brazilian_lps(detections) # Brazilian license plates have seven characters (three letters followed by four digits)
		
		if show_img:
			try:
				print('{0} - {1} characters found:'.format(img_path, str(len(detections))))
				for label, detection in zip(prediction, detections):
					confidence = detection[1]
					x1, y1, x2, y2 = detection[2:6]
					cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1) # draw the bounding box of the character

					print('  {0}:{1:6.2f}% (x1:{2:4} y1:{3:4} w:{4:4} h:{5:4})'.format(label, confidence*100, x1, y1, x2-x1, y2-y1))
				cv2.imshow('{0} - {1}'.format(os.path.basename(img_path), prediction), img)
				cv2.waitKey(3000)
				cv2.destroyAllWindows()
				print()
			except Exception as e:
				print('Unable to show image: ' + str(e))

		result_str = '{0} {1} {2:8}'.format(idx, os.path.abspath(img_path), prediction)

		confidences = ''
		for detection in detections:
			confidences += '{0:.4f} '.format(detection[1])
		result_str += confidences

		if not show_img:
			print(result_str)

		results.append(result_str + '\n')

	with open(results_file, 'w') as f:
		f.writelines(results)
	f.close()

if __name__ == '__main__':
	time1 = time.time()
	main()
	time2 = time.time()
	seconds = (time2-time1)
	m, s = divmod(seconds, 60)
	h, m = divmod(m, 60)

	print('\nTotal time: %d:%02d:%02d' % (h, m, s))