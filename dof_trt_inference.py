from dof_trt_engine import SixDofEnd2End
import numpy as np
import cv2
import time
import os, sys
import argparse
from loguru import logger
from load_process import LoadRealSense, LoadImages
from dof_trt_utils import draw_img

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--yolo_engine", default='./end2end_onnx_2.trt', help="YOLO TRT engine Path")
	parser.add_argument("--dof_engine", default='./sixdresnet.trt', help="DoF TRT engine Path")
	parser.add_argument("--source", help="image path or video path or camera index or realsense")
	parser.add_argument("--center", default=False, action="store_true", help="use center of box, (default: False)")
	parser.add_argument("--box-margin", default=10, help="box margin")
	parser.add_argument("-o", "--output", default="output_trt.png",help="image output path")
	parser.add_argument("-l", "--log", default="./infer_end2end.log",help="log path")
	parser.add_argument("--print-log", default=False, action="store_true",
					help="use end2end engine")
	parser.add_argument('--get-fps', default=False, action="store_true", help='get fps (default: False)')
	parser.add_argument('--show-img', default=False, action="store_true", help='show img (default: False)')
	parser.add_argument('--iter', default=100, type=int, help='get fps (default: 100)')

	args = parser.parse_args()
	print(args)

	logger.add(args.log)
	logger.info(args)
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
			img_preprocess_list = []
			engine_forward_list = []
			t0 = time.perf_counter()
			for i in range(args.iter):
				img = cv2.imread(args.source)
				img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
				t1 = time.perf_counter()
				output = engine.forward(img_rgb)
				t2 = time.perf_counter()
				img_preprocess_list.append(t1-t0)
				engine_forward_list.append(t2-t1)
				t0 = time.perf_counter()
			end_time = time.perf_counter()
			print(f"inference time (iter {args.iter} mean) : {(end_time-start_time)/args.iter:.5f}s ({1/((end_time-start_time)/args.iter)} FPS)")
			print(f"img preprocess time (iter {args.iter} mean) : {np.mean(img_preprocess_list):.5f}s")
			print(f"engine forward time (iter {args.iter} mean) : {np.mean(engine_forward_list):.5f}s")
		else:
			img = cv2.imread(args.source)
			img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			output = engine.forward(img_rgb)
		print(f" # of output : {len(output)}, resized-img shape {output[0].shape}, box info : {output[1]}, \nface shape : {output[4].shape}, p, y, r : {output[5], output[6], output[7]}")
	
	elif args.stream == 'video':
		dataset = LoadImages(args.source, img_size=engine.yolo_model.imgsz[0], center=args.center)
		if args.get_fps:
			img_preprocess_list = []
			engine_forward_list = []
			start_time = time.perf_counter()
			t0 = start_time

			for path, img, im0s, cap in dataset:
				t1 = time.perf_counter()
				output = engine.forward(img)
				t2 = time.perf_counter()
				img_preprocess_list.append(t1-t0)
				engine_forward_list.append(t2-t1)
				t0 = time.perf_counter()
				logger.info(f" # of output : {len(output)}, resized-img shape {output[0].shape}, box info : {output[1]}, \nface shape : {output[4].shape}, p, y, r : {output[5], output[6], output[7]}")
				if cv2.waitKey(1) == ord('q'):  # q to quit
					raise StopIteration
			
			end_time = time.perf_counter()
			print(f"total time : {(end_time-start_time):.5f}s ({1/(end_time-start_time):.5f} FPS)")
			print(f"inference time (iter {len(img_preprocess_list)} mean) : {(end_time-start_time)/len(img_preprocess_list):.5f}s ({1/((end_time-start_time)/len(img_preprocess_list))} FPS)")
			print(f"img preprocess time (iter {len(img_preprocess_list)} mean) : {np.mean(img_preprocess_list):.5f}s")
			print(f"engine forward time (iter {len(img_preprocess_list)} mean) : {np.mean(engine_forward_list):.5f}s")
		else:
			for path, img, im0s, cap in dataset:
				output = engine.forward(img)
				logger.info(f" # of output : {len(output)}, resized-img shape {output[0].shape}, box info : {output[1]}, \nface shape : {output[4].shape}, p, y, r : {output[5], output[6], output[7]}")
				if output is not None:
					if args.show_img:
						out_img = draw_img(output)
						cv2.imshow('img', out_img)
				if cv2.waitKey(1) == ord('q'):
					cv2.destroyAllWindows()
					cap.release()
					raise StopIteration

	elif args.stream == 'cam':
		pass

	elif args.stream == 'rs':
		rs_stream = LoadRealSense(args.source, img_size=engine.yolo_model.imgsz[0])
		for source, tensor_img, preprocess_img ,ori_img, depth_img, depth_img0, _ in rs_stream:
			output = engine.forward(tensor_img)
			if output is not None:
				#logger.info(f" # of output : {len(output)}, resized-img shape {output[0].shape}, box info : {output[1]}, \nface shape : {output[4].shape}, p, y, r : {output[5], output[6], output[7]}")
				if args.show_img:
					out_img = draw_img(output)
					cv2.imshow('rs_img', out_img)
			else:
				if args.show_img:
					cv2.imshow('rs_img', preprocess_img) 
					cv2.imwrite('test_img.jpg', preprocess_img)
			if cv2.waitKey(1) == ord('q'):
				cv2.destroyAllWindows()
				raise StopIteration
			