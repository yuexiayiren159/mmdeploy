# detection_onnxruntime_dynamic_raw_ssd.py
_base_ = ['../_base_/base_dynamic.py', '../../_base_/backends/onnxruntime.py']

# MMDeploy 会将这个配置与 _base_ 中的配置合并，当前文件中的设置优先级更高
codebase_config = dict(
    type='mmdet',
    task='ObjectDetection',
    model_type='end2end', # 尝试 'end2end'，并结合标记来获取原始输出
    # 或者尝试移除 model_type，让它回退到更通用的导出逻辑
    # 我们希望 MMDeploy 使用模型的 forward() 方法的直接输出
    # 这可能需要通过 model_wrapper_cfg 来精确控制
    model_wrapper_cfg=dict(
        model_type='MMDetWrapper', # 或者适合检测模型的通用包装器
        target_is_mmdet=True,
        # 尝试指定导出时调用的方法是 bbox_head 的 forward
        # 这部分比较tricky，可能需要查阅MMDeploy具体实现或示例
        # 一个可能的方向是让包装器直接返回 head 的输出
        # 或者通过标记来指定 head 的输出作为最终输出
        output_signature_keys=[ # 确保这些与 TinySSDHead.forward 中 mark 的名称一致
            'cls_score_0', 'bbox_pred_0',
            'cls_score_1', 'bbox_pred_1',
            'cls_score_2', 'bbox_pred_2',
            'cls_score_3', 'bbox_pred_3',
            'cls_score_4', 'bbox_pred_4'
        ]
    )
)

onnx_config = dict(
    # output_names 需要与 TinySSDHead.forward 中 mark 的名称一致
    output_names=[
        'cls_score_0', 'bbox_pred_0',
        'cls_score_1', 'bbox_pred_1',
        'cls_score_2', 'bbox_pred_2',
        'cls_score_3', 'bbox_pred_3',
        'cls_score_4', 'bbox_pred_4'
    ],
    # dynamic_axes 也需要确保输出的维度是正确的
    # 假设每个 cls_score 是 (batch, num_classes, H, W)
    # 假设每个 bbox_pred 是 (batch, num_anchors_per_loc * 4, H, W)
    # 或者根据实际情况调整
    dynamic_axes={
        'input': {0: 'batch', 2: 'height', 3: 'width'},
        'cls_score_0': {0: 'batch'}, 
        'bbox_pred_0': {0: 'batch'},
        'cls_score_1': {0: 'batch'},
        'bbox_pred_1': {0: 'batch'},
        'cls_score_2': {0: 'batch'},
        'bbox_pred_2': {0: 'batch'},
        'cls_score_3': {0: 'batch'},
        'bbox_pred_3': {0: 'batch'},
        'cls_score_4': {0: 'batch'},
        'bbox_pred_4': {0: 'batch'},
    }
)

# 由于我们已经在 TinySSDHead.forward 中添加了 @mark，
# 并且 output_names 与标记名匹配，MMDeploy 应该能够识别这些作为期望的输出。
# 如果这不起作用，下一步是使用 partition_config 来更明确地指定结束点。
# deploy_cfg = dict(
#     partition_config=dict(
#         apply_marks=True,
#         partition_cfg=[
#             dict(
#                 save_file='end2end.onnx', # 与 torch2onnx.sh 中的 --work-dir 匹配
#                 # start_marker 需要标记模型的输入，例如在 TinySSDDetector.forward 的输入上
#                 # start_marker=['input'], # 假设模型输入被标记为 'input'
#                 end_marker=[ # 确保这些与 TinySSDHead.forward 中 mark 的名称一致
#                     'cls_score_0', 'bbox_pred_0',
#                     'cls_score_1', 'bbox_pred_1',
#                     'cls_score_2', 'bbox_pred_2',
#                     'cls_score_3', 'bbox_pred_3',
#                     'cls_score_4', 'bbox_pred_4'
#                 ]
#             )
#         ])
# )
