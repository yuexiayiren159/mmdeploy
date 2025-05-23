_base_ = ['./detection_onnxruntime_static.py']

onnx_config = dict(input_shape=[320, 320])
partition_config = dict(
    type='yolov3_partition',
    apply_marks=True,
    partition_cfg=[
        dict(
            save_file='yolov3.onnx',
            start=['detector_forward:input'],
            end=['yolo_head:input'],
            output_names=[f'pred_maps.{i}' for i in range(3)])
    ])
