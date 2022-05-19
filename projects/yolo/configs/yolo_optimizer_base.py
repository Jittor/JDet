parameter_groups_generator = dict(
    type='YoloParameterGroupsGenerator',
    weight_decay=0.0005, #hyp[weight_decay]
)
optimizer=dict(
    type='SGD',
    lr=0.01, # hyp[lr0]
    momentum=0.937, #hyp[momentum]
    nesterov=True
)