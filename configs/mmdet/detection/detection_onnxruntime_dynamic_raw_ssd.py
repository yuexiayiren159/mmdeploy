# ./configs/mmdet/detection/detection_onnxruntime_dynamic_raw_ssd.py
_base_ = ['../_base_/base_dynamic.py', '../../_base_/backends/onnxruntime.py']

# 1. 代码库配置 (Codebase Config)
codebase_config = dict(
    type='mmdet',
    task='ObjectDetection',
    # 我们希望 MMDeploy 使用模型的 forward() 方法的直接输出
    # 通过标记来精确控制输出
)

# 2. ONNX 配置 (ONNX Config)
onnx_config = dict(
    type='onnx',
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=11,
    save_file='end2end.onnx', 
    input_shape=None, 
    input_names=['input'], 
    output_names=[        
        'cls_score_0', 'bbox_pred_0',
        'cls_score_1', 'bbox_pred_1',
        'cls_score_2', 'bbox_pred_2',
        'cls_score_3', 'bbox_pred_3',
        'cls_score_4', 'bbox_pred_4'
    ],
    dynamic_axes={
        'input': {
            0: 'batch',
            2: 'height',
            3: 'width'
        },
        'cls_score_0': {0: 'batch'}, 'bbox_pred_0': {0: 'batch'},
        'cls_score_1': {0: 'batch'}, 'bbox_pred_1': {0: 'batch'},
        'cls_score_2': {0: 'batch'}, 'bbox_pred_2': {0: 'batch'},
        'cls_score_3': {0: 'batch'}, 'bbox_pred_3': {0: 'batch'},
        'cls_score_4': {0: 'batch'}, 'bbox_pred_4': {0: 'batch'},
    })

# 3. 部署配置 (Deploy Config / Backend Config)
# 确保模型代码 (ssd_custom_head.py 中的 TinySSDHead.forward)
# 正确地使用了 @mark(output_name, tensor_to_mark)，
# 并且 onnx_config.output_names 与这些标记名一致。
