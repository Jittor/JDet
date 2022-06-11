model = dict(
    ch=3, 
    nc=80,
    boxlg=0.05, # box loss gain
    clslg=0.5,  # class loss gain  
    objlg=1.0,  # object loss gain
    cls_pw=1.0, # class positive weight
    obj_pw=1.0, # object positive weight
    fl_gamma=0.0,   # focal loss gamma
    anchor_t=4.0,  # anchor threshold
)