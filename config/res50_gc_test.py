backbone=dict(
    type='ResNet',
    depth=50,
    style='pytorch',
    stage_with_context_block=[False, True, True, True],
    context_block_cfg=dict(ratio=1./4)
)