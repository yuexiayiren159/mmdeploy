# _base_ = ['./detection_onnxruntime_static.py']
_base_ = ['../../_base_/backends/onnxruntime.py']
# backend_config = dict(type='onnxruntime')


codebase_config = dict(
    type='mmdet',
    task='ObjectDetection',
    model_type='end2end',
    post_processing=dict(
        score_threshold=0.05,
        confidence_threshold=0.005,  # for YOLOv3
        iou_threshold=0.5,
        max_output_boxes_per_class=200,
        pre_top_k=5000,
        keep_top_k=100,
        background_label_id=-1,
    ))

onnx_config = dict(
    type='onnx',
    export_params=True,
    keep_initializers_as_inputs=True,
    opset_version=11,
    save_file='end2end.onnx',
    input_names=['input'],
    output_names=['dets', 'labels'],
    input_shape=[736, 416],
    optimize=True)
partition_config = dict(
    type='yolov3_partition',
    apply_marks=True,
    partition_cfg=[
        dict(
            save_file='yolov3_xceptionb0.onnx',
            start=['detector_forward:input'],
            end=['yolo_head:input'],
            output_names=[f'pred_maps.{i}' for i in range(3)])
    ])
