Requirements
	The Darknet framework is self-contained in the "darknet" folder and must be compiled before running the tests. To build Darknet just type "make" in the "darknet" folder.
	The current version was tested on a machine running Ubuntu 18.04, with OpenCV 3.4.1, NumPy 1.14.3 and Python 3.5.2.

Running (Linux)
	After building the Darknet framework, you must execute the "cnn-ocr.py" script to run CNN-OCR ("python3 cnn-ocr.py"). It has 7 optional arguments:
	(-p) path of the input images [default = './input-images/']
  	(-c) cfg file [default = './cfg/cr-net.cfg']
  	(-w) weights file [default = './weights/cr-net.weights']
  	(-d) data file [default = './data/obj.data']
  	(-bb) add black borders to the license plate patches so that they have an aspect ratio (w/h) between 2.5 and 3.0. In this way, the network processes less distorted images (the aspect ratio of the input image is 2.75) [default = True]
  	(-r) path of the results file [default = \'./output.txt\']
  	(-s) compute (and show) the bounding boxes [default = False]

A word on GPU and CPU
	We know that not everyone has a high-end GPU available and that sometimes it is cumbersome to properly configure CUDA. 
	Therefore, we chose to set Darknet's makefile to use CPU as default instead of GPU to favor easy execution for most people rather than fast performance.
	If you want to speed up the process, please edit the variables in Darknet's makefile to use GPU.