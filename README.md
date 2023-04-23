## YOLO+DoF End2End model

### First, prepare tensorRT model YOLO and SixDoF
#### 1.1 Generate TRT YOLO model
YOLO v7 link : https://github.com/pdh930105/yolov7-face

```
git clone https://github.com/pdh930105/yolov7-face.git
cd yolov7-face
python3 models/export.py --weights yolov7-tiny-face.pt --grid --simplify
python3 models/export_tensorrt.py -o yolov7-tiny-face_wo_nms.onnx -e end2end_yolo.trt --end2end --max_det 10 --rkpts # using onnx
```

#### 1.2 Generate TRT SixDRepNet (base model : resnet18_dof)
SixDof model link : https://github.com/pdh930105/6DRepNet.git
```
git clone https://github.com/pdh930105/6DRepNet.git
cd 6DRepNet
python3 sixdrepnet/export_tensorrt.py -o resnet18_dof.onnx -e end2end_dof.trt
```

### Second, If prepared TensorRT model, run end2end model
```
cd dof-inference
python3 dof_trt_inference.py --yolo_engine end2end_yolo.trt --dof_engine end2end_dof.trt --source obama.jpg 
# --source 'cam' = webcam (future works), 'rgb' = realsense's rgb, 'ir'=realsense's infrared
# --show-img : visualize video/webcam/realsense
# --get-fps : visualize FPS time
# --iter : iteration for calculating FPS time
```

