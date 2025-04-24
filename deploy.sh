#/root/lanyun-tmp/download/model/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth


#python ./tools/deploy.py \
#    ./configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
#    ../mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py \
#    /root/lanyun-tmp/download/model/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
#    ../mmdetection/demo/demo.jpg \
#    --work-dir mmdeploy_model/faster-rcnn \
#    --device cuda \
#    --dump-info

# python ./tools/deploy.py \
#     configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
#     $PATH_TO_MMDET/configs/yolo/yolov3_d53_8xb8-ms-608-273e_coco.py \
#     $PATH_TO_MMDET/checkpoints/yolo/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth \
#     $PATH_TO_MMDET/demo/demo.jpg \
#     --work-dir work_dir \
#     --show \
#     --device cuda:0

# python ./tools/deploy.py \
#     ./configs/mmdet/detection/detection_onnxruntime_dynamic.py \
#     ../mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py \
#     /root/lanyun-tmp/download/model/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
#     ../mmdetection/demo/demo.jpg \
#     --work-dir mmdeploy_model/faster-rcnn \
#     --device cpu \
#     --dump-info

# python ./tools/deploy.py \
#     ./configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
#     ../mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py \
#     /root/lanyun-tmp/download/model/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
#     ../mmdetection/demo/demo.jpg \
#     --work-dir mmdeploy_model/faster-rcnn \
#     --device cpu \
#     --dump-info

# Pytorch转onnx
# python tools/deploy.py \
#         configs/mmseg/segmentation_onnxruntime_dynamic.py \
#         ../mmsegmentation/Zihao-Configs/ZihaoDataset_FastSCNN_20230818.py \
#         ../mmsegmentation/checkpoint/FastSCNN/Zihao_FastSCNN.pth \
#         ../mmsegmentation/demo/dua-hau-1.jpg \
#         --work-dir mmdeploy_model/mmseg2onnx_fastscnn \
#         --dump-info

# 模型转换-Pytorch转ONNX（静态batch）
# python tools/deploy.py \
#         configs/mmseg/segmentation_onnxruntime_static-1024x2048.py \
#         ../mmsegmentation/Zihao-Configs/ZihaoDataset_FastSCNN_20230818.py \
#         ../mmsegmentation/checkpoint/FastSCNN/Zihao_FastSCNN.pth \
#         ../mmsegmentation/demo/dua-hau-1.jpg \
#         --work-dir mmdeploy_model/mmseg2onnx_fastscnn_static \
#         --dump-info

# Pytorch转NCNN
python tools/deploy.py \
        configs/mmseg/segmentation_ncnn_static-512x512.py \
        ../mmsegmentation/Zihao-Configs/ZihaoDataset_FastSCNN_20230818.py \
        ../mmsegmentation/checkpoint/FastSCNN/Zihao_FastSCNN.pth \
        ../mmsegmentation/demo/dua-hau-1.jpg \
        --work-dir mmdeploy_model/mmseg2ncnn_fastscnn \
        --dump-info