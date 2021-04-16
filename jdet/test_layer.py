import numpy as np

def test_jittor():
    import jittor as jt
    import layers.roi_align
    import layers.psroi_align
    jt.flags.use_cuda = 1
    # model = layers.roi_align.ROIAlign(output_size=1, spatial_scale=1.0, sampling_ratio=-1)
    model = layers.psroi_align.PSROIAlign(output_size=5, spatial_scale=1.0, sampling_ratio=-1, out_dim=4)
    input = np.load('test.npy') # np.random.rand([1, 100, 50, 50])
    input = jt.float32(input)
    roi = np.array([[0, 10, 10, 20, 20], [0, 15, 15, 25, 25], [0, 27, 10, 36, 20]])
    roi = jt.float32(roi)
    output = model(input, roi)
    print(output)

def test_pytorch():
    import torch
    import torchvision
    input = np.load('test.npy') # np.random.rand([1, 100, 50, 50])
    input = torch.tensor(input).float()
    print(input.shape)
    roi = np.array([[0, 10, 10, 20, 20], [0, 15, 15, 25, 25], [0, 27, 10, 36, 20]])
    roi = torch.tensor(roi).float()
    output = torchvision.ops.ps_roi_align(input=input, boxes=roi, output_size=(5, 5))
    print(output)

if __name__ == "__main__":
    test_jittor()
    # test_pytorch()