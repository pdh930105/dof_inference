from dof_trt_engine import SixDofEnd2End
import numpy as np
import cv2
import time
import os, sys
import argparse
from loguru import logger
from load_process import LoadRealSense, LoadImages
from dof_trt_utils import draw_img

import socket
import time


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
	parser.add_argument('--server', action='store_true', help="activate server")
	parser.add_argument('--ip-addr', default='localhost', help="required socket ip (local ip addr)")
    
	args = parser.parse_args()
	print(args)

	if args.server == True:
		# opt.server_host = "163.152.172.78"
		assert args.ip_addr !='localhost', "required ip_addr for using --server option"
		args.server_host = args.ip_addr
		args.server_port = 10243
		args.servers = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		args.servers.bind((args.server_host, args.server_port))
		print("Waiting for client to connect...")
		args.servers.listen(1)
		args.server_conn, args.server_addr = args.servers.accept()
		print('Connected by ', args.server_addr)

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
			yolo_preprocess_list = []
			yolo_infer_list = []
			dof_preprocess_list = []
			dof_infer_list = []
			engine_forward_list = []
			t0 = time.perf_counter()
			for i in range(args.iter):
				img = cv2.imread(args.source)
				img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
				t1 = time.perf_counter()
				yolo_preproc_img = engine.yolo_img_preprocess(img_rgb)
				h, w = yolo_preproc_img.shape[1:]
				t2 = time.perf_counter()
				detect_result, box_detected = engine.yolo_model.forward_yolo(yolo_preproc_img)
				t3 = time.perf_counter()
				if box_detected:
					detect_box = detect_result[:4]
					detect_score = detect_result[4] * detect_result[5]
					detect_kp = detect_result[6:]
					x1, x2, y1, y2 = detect_box[0] - detect_box[2]/2, detect_box[0] + detect_box[2]/2, detect_box[1] - detect_box[3]/2, detect_box[1] + detect_box[3]/2
					detect_box_xyxy = [x1, y1, x2, y2]
					face_img = yolo_preproc_img.transpose(1,2,0)[max(int(y1)-engine.box_margin, 0):min(int(y2)+engine.box_margin, h), max(int(x1)-engine.box_margin, 0):min(int(x2)+engine.box_margin, w), :]
					t4 = time.perf_counter()
					pitch, yaw, roll = engine.dof_model.dof_forward(face_img)
					t5 = time.perf_counter()
				
					img_preprocess_list.append(t1-t0)
					yolo_preprocess_list.append(t2-t1)
					yolo_infer_list.append(t3-t2)
					dof_preprocess_list.append(t4-t3)
					dof_infer_list.append(t5-t4)
					engine_forward_list.append(t5-t0)
					output= [yolo_preproc_img, detect_box, detect_score, detect_kp, face_img, pitch, yaw, roll]
				t0 = time.perf_counter()

			end_time = time.perf_counter()
			print(f"inference time (iter {args.iter} mean) : {(end_time-start_time)/args.iter:.5f}s ({1/((end_time-start_time)/args.iter)} FPS)")
			print(f"img preprocess time (iter {args.iter} mean) : {np.mean(img_preprocess_list):.5f}s")
			print(f"yolo preprocess time (iter {args.iter} mean) : {np.mean(yolo_preprocess_list):.5f}s")
			print(f"yolo infer time (iter {args.iter} mean) : {np.mean(yolo_infer_list):.5f}s")
			print(f"dof preprocess time (iter {args.iter} mean) : {np.mean(dof_preprocess_list):.5f}s")
			print(f"dof infer time (iter {args.iter} mean) : {np.mean(dof_infer_list):.5f}s")
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
				#logger.info(f" # of output : {len(output)}, resized-img shape {output[0].shape}, box info : {output[1]}, \nface shape : {output[4].shape}, p, y, r : {output[5], output[6], output[7]}")
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
				if args.server == True:
					data = f'{float(output[5]):.3f},{float(output[6]):.3f},{float(output[7]):.3f},{int(output[3][6])},{int(output[3][7])},'
					try:
						args.server_conn.send(data.encode())
					except Exception as e:
						print(data), print(e)
				else:
					pass
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
			