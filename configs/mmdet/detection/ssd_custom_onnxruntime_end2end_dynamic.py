# ./configs/mmdet/detection/ssd_custom_onnxruntime_end2end_dynamic.py (你创建或修改的文件)

_base_ = [
    '../../_base_/backends/onnxruntime.py',             # 通用ONNX Runtime后端配置
    '../_base_/base_dynamic.py',                  # 通用动态部署配置
    # '../../_base_/deployments/base_detection_dynamic.py' # 通用动态检测部署配置 (可能包含动态轴等)
                                                        # 或者一个静态配置如 base_detection_static.py
]

codebase_config = dict(
    type='mmdet',
    task='ObjectDetection',
    model_type='end2end',  # <--- 关键：'end2end' 通常表示包含后处理
    post_processing=dict( # <--- 后处理参数，确保这些与你训练和评估时的设置一致
        score_threshold=0.05,
        iou_threshold=0.5,
        max_output_boxes_per_class=200, # YOLOv3的这个参数可能与SSD不同，SSD通常是keep_top_k
        pre_top_k=-1, # SSD中可能没有这个，或者有nms_pre
        keep_top_k=100, # SSD中NMS后保留的框数量
        background_label_id=-1 # 对于SSD，如果num_classes=1, cls_out_channels=2, 背景类是1
                               # MMDetection的后处理通常会自动处理
    )
)

onnx_config = dict(
    # opset_version 通常在 _base_/backends/onnxruntime.py 中定义，可以不在这里覆盖
    # save_file='end2end.onnx', # 输出文件名会由 --work-dir 和 deploy.py 自动生成
    input_names=['input'],
    output_names=['dets', 'labels'], # <--- 关键：后处理后的输出名称
    input_shape=None, # 设置为None以使用动态输入，或指定如 [1, 3, 320, 320]
    dynamic_axes={ # 如果 input_shape=None，则需要定义动态轴
        'input': {
            0: 'batch',
            2: 'height',
            3: 'width'
        },
        'dets': {
            0: 'batch',
            1: 'num_dets' # 检测框数量是动态的
        },
        'labels': {
            0: 'batch',
            1: 'num_dets' # 标签数量与检测框数量一致
        }
    },
    optimize=True # 进行ONNX图优化
)

# 如果你的SSD模型有特定于模型的部署参数，可以在这里覆盖或添加
# model_cfg = dict(...)