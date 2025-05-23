# 强制清除相关的 Python 缓存
# echo "Attempting to clear Python cache..."
# rm -rf ./mmdeploy/codebase/mmdet/models/dense_heads/__pycache__
# rm -f ./mmdeploy/codebase/mmdet/models/dense_heads/*.pyc
# rm -rf ../mmdetection/mmdet/models/dense_heads/__pycache__
# rm -f ../mmdetection/mmdet/models/dense_heads/*.pyc
# echo "Python cache clearing attempt finished."

# 执行转换命令，实现端到端的转换
#python ./tools/deploy.py \
#    ./configs/mmdet/detection/detection_onnxruntime_dynamic.py \
#    ../mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py \
#    /root/lanyun-tmp/download/model/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
#    ../mmdetection/demo/demo.jpg \
#    --work-dir mmdeploy_model/faster-rcnn \
#    --device cpu \
#    --dump-info

#python ./tools/torch2onnx.py \
#  ./configs/mmdet/detection/detection_onnxruntime_dynamic.py \
#  ../mmdetection/configs/yolo/yolov3_mobilenetv2_8xb24-320-300e_coco_my_2.py \
#  /root/lanyun-tmp/my_work/mmdetection/yolov3_mobilenetv2_8xb24-320-300e_coco_my_1/epoch_35.pth \
#  ../mmdetection/demo/demo.jpg \
#  --work-dir mmdeploy_model/yolov3_mobilenetv2 \
#  --device cpu

#python tools/torch2onnx.py \
#  configs/mmdet/detection/yolov3_partition_onnxruntime_static.py \
#  ../mmdetection/configs/yolo/yolov3_d53_8xb8-ms-608-273e_coco.py \
#  https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_mstrain-608_273e_coco/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth \
#  ../mmdetection/demo/demo.jpg \
#  --work-dir mmdeploy_model/yolov3 \
#  --device cpu


# python ./tools/torch2onnx.py \
#   ./configs/mmdet/detection/yolov3_partition_onnxruntime_static_320.py \
#   ../mmdetection/configs/yolo/yolov3_mobilenetv2_8xb24-320-300e_coco_my_1.py \
#   /root/lanyun-tmp/my_work/mmdetection/yolov3_mobilenetv2_8xb24-320-300e_coco_my_1/epoch_35.pth \
#   ../mmdetection/demo/demo.jpg \
#   --work-dir mmdeploy_model/yolov3_mobilenetv2 \
#   --device cpu


# python tools/torch2onnx.py \
#  configs/mmdet/detection/yolov3_partition_onnxruntime_static.py \
#  ../mmdetection/configs/yolo/yolov3_d53_8xb8-ms-608-273e_coco.py \
#  https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_mstrain-608_273e_coco/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth \
#  ../mmdetection/demo/demo.jpg \
#  --work-dir mmdeploy_model/yolov3 \
#  --device cpu


#  python tools/torch2onnx.py \
#  configs/mmdet/detection/yolov3_xceptionb0_partition_onnxruntime_static_416-736.py \
#  ../mmdetection/configs/yolo/yolov3_xceptionb0_1xb24-416-736-coco256.py \
#  ../mmdetection/work_dirs/yolov3_xceptionb0_1xb24-416-736-coco256/epoch_1.pth \
#  ../mmdetection/demo/demo.jpg \
#  --work-dir mmdeploy_model/yolov3_xceptionb0 \
#  --device cpu



# python ./tools/torch2onnx.py \
#   ./configs/mmdet/detection/yolov3_partition_onnxruntime_static_416.py \
#   ../mmdetection/configs/yolo/yolov3_mobilenetv2_8xb24-ms-416-300e_coco_my.py \
#   /root/lanyun-tmp/my_work/mmdetection/yolov3_mobilenetv2_8xb24-ms-416-300e_coco_my/epoch_4.pth \
#   ../mmdetection/demo/demo.jpg \
#   --work-dir mmdeploy_model/yolov3_mobilenetv2 \
#   --device cpu

# 分割模型fastscnn
# python ./tools/torch2onnx.py \
#   ./configs/mmseg/segmentation_onnxruntime_static-512x512.py \
#   ../mmsegmentation/configs/fastscnn/LaneDataset_FastSCNN_20230818.py \
#   /root/lanyun-tmp/openmmlab/mmsegmentation/work_dirs/LaneDataset-FastSCNN/iter_50000.pth \
#   /root/lanyun-tmp/openmmlab/mmsegmentation/Lane_data/img_dir/val/road18_MAY_9_85.png \
#   --work-dir mmdeploy_model/LaneDataset_FastSCNN \
#   --device cpu

# python ./tools/torch2onnx.py \
#   ./configs/mmseg/segmentation_onnxruntime_static-416x416_my.py \
#   ../mmsegmentation/configs/mobilenet_v2/Mobilenetv2_PSPNet-lane_20240716.py \
#   /root/lanyun-tmp/openmmlab/mmsegmentation/work_dirs/Mobilenetv2-PSPNet-lane/best_mIoU_iter_39500.pth \
#   /root/lanyun-tmp/openmmlab/mmsegmentation/Lane_data/img_dir/val/road18_MAY_9_85.png \
#   --work-dir mmdeploy_model/Mobilenetv2_PSPNet-lane \
#   --device cpu


#python ./tools/torch2onnx.py \
#  ./configs/mmseg/segmentation_onnxruntime_static-512x512.py \
#  ../mmyolo/configs/yolov8/yolov8_n_syncbn_fast_8xb16-500e_coco.py \
#  /root/lanyun-tmp/openmmlab/mmsegmentation/work_dirs/LaneDataset-FastSCNN/iter_50000.pth \
#  /root/lanyun-tmp/openmmlab/mmsegmentation/Lane_data/img_dir/val/road18_MAY_9_85.png \
#  --work-dir mmdeploy_model/LaneDataset_FastSCNN \
#  --device cpu

# python ./tools/torch2onnx.py \
#   ./configs/mmdet/detection/yolov3_partition_onnxruntime_static_416.py \
#   ../mmdetection/configs/yolo/yolov3_mobilenetv2_8xb24-ms-416-300e_coco_my.py \
#   /root/lanyun-tmp/my_work/mmdetection/yolov3_mobilenetv2_8xb24-ms-416-300e_coco_my/epoch_4.pth \
#   ../mmdetection/demo/demo.jpg \
#   --work-dir mmdeploy_model/yolov3_mobilenetv2 \
#   --device cpu


#  python ./tools/torch2onnx.py \
#    ./configs/mmdet/detection/yolov3_partition_onnxruntime_static_288_416.py \
#    ../mmdetection/configs/yolo/yolov3_mobilenetv2_8xb24-ms-416-300e_coco_my.py \
#    E:\workspace\lanyun_work\openmmlab\mmdetection\checkpoints\yolov3_mobilenetv2_mstrain-416_300e_coco_20210718_010823-f68a07b3.pth \
#    ../mmdetection/demo/demo.jpg \
#    --work-dir mmdeploy_model/yolov3_mobilenetv2 \
#    --device cpu


#  python ./tools/torch2onnx.py \
#    ./configs/mmdet/detection/single-stage_ncnn_static-320x320.py \
#    E:/workspace/openmmlab/mmdetection/my_configs/ssdlite_mobilenetv2-scratch_8xb24-600e_coco.py \
#    E:/workspace/openmmlab/mmdetection/work_dirs/ssdlite_mobilenetv2-scratch_8xb24-600e_coco/epoch_35.pth \
#    ../mmdetection/demo/demo.jpg \
#    --work-dir mmdeploy_model/ssdlite_mobilenetv2-scratch_8xb24-600e_coco \
#    --device cpu

  
# configs\mmdet\detection\detection_onnxruntime_static.py
#  python ./tools/torch2onnx.py \
#    ./configs/mmdet/detection/detection_onnxruntime_static.py \
#    E:/workspace/openmmlab/mmdetection/my_configs/ssdlite_mobilenetv2-scratch_8xb24-600e_coco.py \
#    E:/workspace/openmmlab/mmdetection/work_dirs/ssdlite_mobilenetv2-scratch_8xb24-600e_coco/epoch_35.pth \
#    ../mmdetection/demo/demo.jpg \
#    --work-dir mmdeploy_model/ssdlite_mobilenetv2-scratch_8xb24-600e_coco \
#    --device cpu


#  python ./tools/torch2onnx.py \
#    ./configs/mmdet/detection/single-stage_ncnn_static-320x320.py \
#    E:/workspace/openmmlab/mmdetection/my_configs/ssdlite_mobilenetv2-scratch_8xb24-600e_coco.py \
#    E:/workspace/openmmlab/mmdetection/work_dirs/ssdlite_mobilenetv2-scratch_8xb24-600e_coco/epoch_35.pth \
#    ../mmdetection/demo/demo.jpg \
#    --work-dir mmdeploy_model/ssdlite_mobilenetv2-scratch_8xb24-600e_coco \
#    --device cpu

# 测试mmcls转换onnx
# python ./tools/deploy.py \
#   ./configs/mmpretrain/classification_onnxruntime_dynamic.py \
#   ./my_work/mmclass_demo/mobilenet_v2_1x_fruit30.py \
#   ./my_work/mmclass_demo/fruit30_mmcls.pth \
#   ./my_work/mmclass_demo/demo.JPEG \
#   --work-dir mmdeploy_model/mmcls/fruit30_mmcls \
#   --device cpu \
#   --dump-info


# run the command to start model conversion
# python ./tools/deploy.py \
#     ./configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
#     ../mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py \
#     ./my_work/mmclass_demo/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
#     ../mmdetection/demo/demo.jpg \
#     --work-dir mmdeploy_model/faster-rcnn_1 \
#     --device cuda \
#     --dump-info


# yolox
# python ./tools/deploy.py \
#         ./configs/mmdet/detection/detection_onnxruntime_dynamic.py \
#         ../mmdetection//configs/yolox/yolox_tiny_8xb8-300e_coco.py \
#         ./mmdeploy_model/yolox/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth \
#         ../mmdetection/demo/demo.jpg \
#         --work-dir ./mmdeploy_model/yolox/yolox \
#         --device cpu \
#         --dump-info




# ssd-custom
# python ./tools/deploy.py \
#         ./configs/mmdet/detection/single-stage_ncnn_static-256x256_my.py \
#         ../mmdetection//configs/ssd/ssd_custom.py \
#         ./mmdeploy_model/ssd/epoch_12.pth \
#         ../mmdetection/data/banana-coco/train2017/0.png \
#         --work-dir ./mmdeploy_model/ssd_1 \
#         --device cpu \
#         --dump-info

# ssd不带后处理  出现错误
# python -I ./tools/deploy.py \
#         ./configs/mmdet/detection/detection_onnxruntime_dynamic_raw_ssd.py \
#         ../mmdetection//configs/ssd/ssd_custom.py \
#         ../mmdetection/work_dirs/ssd_custom/epoch_12.pth \
#         ../mmdetection/data/banana-coco/train2017/0.png \
#         --work-dir ./mmdeploy_model/ssd_2 \
#         --device cpu \
#         --dump-info

# ssd-custom
# python ./tools/deploy.py \
#         ./configs/mmdet/detection/single-stage_ncnn_static-256x256_my.py \
#         ../mmdetection//configs/ssd/ssd_custom_export.py \
#         ./mmdeploy_model/ssd/epoch_12.pth \
#         ../mmdetection/data/banana-coco/train2017/0.png \
#         --work-dir ./mmdeploy_model/ssd_2 \
#         --device cpu \
#         --dump-info

# ssd-custom 包含后处理  成功导出，而且onnx模型也能推理
# python tools/deploy.py \
#     configs/mmdet/detection/ssd_custom_onnxruntime_end2end_dynamic.py \
#     ../mmdetection/configs/ssd/ssd_custom.py \
#     ../mmdetection/work_dirs/ssd_custom/epoch_12.pth \
#     ../mmdetection/data/banana-coco/train2017/0.png \
#     --work-dir work_dirs/ssd_custom_onnx_end2end \
#     --device cpu \
#     --show \
#     --dump-info


# 导出没有后处理的模型
#  python ./tools/torch2onnx.py \
#    ./configs/mmdet/detection/yolov3_partition_onnxruntime_static_288_416.py \
#    ../mmdetection/configs/yolo/yolov3_mobilenetv2_8xb24-ms-416-300e_coco_my.py \
#    E:\workspace\lanyun_work\openmmlab\mmdetection\checkpoints\yolov3_mobilenetv2_mstrain-416_300e_coco_20210718_010823-f68a07b3.pth \
#    ../mmdetection/demo/demo.jpg \
#    --work-dir mmdeploy_model/yolov3_mobilenetv2 \
#    --device cpu

