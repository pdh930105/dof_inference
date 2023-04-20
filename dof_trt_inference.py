from dof_trt_engine import SixDofEnd2End
import numpy as np
import cv2
import time
import os, sys
import argparse
from loguru import logger
from load_process import LoadRealSense

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--yolo_engine", default='./end2end_onnx_2.trt', help="YOLO TRT engine Path")
	parser.add_argument("--dof_engine", default='./sixdresnet.trt', help="DoF TRT engine Path")
	parser.add_argument("--source", help="image path or video path or camera index or realsense")
	parser.add_argument("--box-margin", default=10, help="box margin")
	parser.add_argument("-o", "--output", default="output_trt.png",help="image output path")
	parser.add_argument("-l", "--log", default="./infer_end2end.log",help="log path")
	parser.add_argument("--print-log", default=False, action="store_true",
					help="use end2end engine")
	parser.add_argument('--get-fps', default=False, action="store_true", help='get fps (default: False)')
	parser.add_argument('--iter', default=100, type=int, help='get fps (default: 100)')

	args = parser.parse_args()
	print(args)

	logger.add(args.log)
	img_type = ['jpg', 'png', 'jpeg', 'bmp']
	video_type = ['mp4', 'avi', 'mov', 'mkv']

	if  args.source.lower().split('.')[-1] in img_type:
		args.stream = 'img'
	elif args.source.lower().split('.')[-1] in video_type:
		args.stream = 'video'
	elif args.source.isdigit():
		args.stream = 'cam'
	elif args.source == 'rgb' or args.source == 'ir':
		args.stream = 'rs'
	else:
		raise ValueError("source is not valid")

	engine = SixDofEnd2End(yolo_engine_path=args.yolo_engine, dof_engine_path=args.dof_engine, box_margin=args.box_margin, stream_type=args.stream, logger=logger, print_log=args.print_log)

	if args.stream == 'img':
		start_time = time.perf_counter()
		if args.get_fps:
			for i in range(args.iter):
				img = cv2.imread(args.source)
				img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
				output = engine.forward(img_rgb)
			end_time = time.perf_counter()
			print(f"inference time (iter {args.iter} mean) : {(end_time-start_time)/args.iter:.5f}s ({1/((end_time-start_time)/args.iter)} FPS)")
		else:
			img = cv2.imread(args.source)
			img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			output = engine.forward(img_rgb)
		print(f" # of output : {len(output)}, resized-img shape {output[0].shape}, box info : {output[1]}, face shape : {output[4].shape}, p, y, r : {output[5], output[6], output[7]}")
	
	elif args.stream == 'video':
		pass

	elif args.stream == 'cam':
		pass

	elif args.stream == 'rs':
		rs_stream = LoadRealSense(args.source, img_size=engine.yolo_model.imgsz[0])
		for source, tensor_img, preprocess_img ,ori_img, depth_img, depth_img0, _ in rs_stream:
			output = engine.forward(tensor_img)
			print(f" # of output : {len(output)}, resized-img shape {output[0].shape}, box info : {output[1]}, face shape : {output[4].shape}, p, y, r : {output[5], output[6], output[7]}")

	"""
	if video:
		use_cam = True if video.isdigit() else False
		if args.v1:
			pred.detect_video(video, conf=0.5, use_cam=use_cam, end2end=args.end2end) # set 0 use a webcam
		else:
			pred.detect_video_v2(video, conf=0.5, use_cam=use_cam, end2end=args.end2end)
	if rs:
		pred.detect_rs(args.rs_type, conf=0.5)
	"""