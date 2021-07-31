import pickle as pk
import argparse
ours_names_ = """backbone.C1.0.weight
backbone.C1.1.weight
backbone.C1.1.bias
backbone.C1.1.running_mean
backbone.C1.1.running_var
backbone.C1.3.weight
backbone.C1.4.weight
backbone.C1.4.bias
backbone.C1.4.running_mean
backbone.C1.4.running_var
backbone.C1.6.weight
backbone.C1.7.weight
backbone.C1.7.bias
backbone.C1.7.running_mean
backbone.C1.7.running_var
backbone.layer1.0.conv1.weight
backbone.layer1.0.bn1.weight
backbone.layer1.0.bn1.bias
backbone.layer1.0.bn1.running_mean
backbone.layer1.0.bn1.running_var
backbone.layer1.0.conv2.weight
backbone.layer1.0.bn2.weight
backbone.layer1.0.bn2.bias
backbone.layer1.0.bn2.running_mean
backbone.layer1.0.bn2.running_var
backbone.layer1.0.conv3.weight
backbone.layer1.0.bn3.weight
backbone.layer1.0.bn3.bias
backbone.layer1.0.bn3.running_mean
backbone.layer1.0.bn3.running_var
backbone.layer1.0.downsample.1.weight
backbone.layer1.0.downsample.2.weight
backbone.layer1.0.downsample.2.bias
backbone.layer1.0.downsample.2.running_mean
backbone.layer1.0.downsample.2.running_var
backbone.layer1.1.conv1.weight
backbone.layer1.1.bn1.weight
backbone.layer1.1.bn1.bias
backbone.layer1.1.bn1.running_mean
backbone.layer1.1.bn1.running_var
backbone.layer1.1.conv2.weight
backbone.layer1.1.bn2.weight
backbone.layer1.1.bn2.bias
backbone.layer1.1.bn2.running_mean
backbone.layer1.1.bn2.running_var
backbone.layer1.1.conv3.weight
backbone.layer1.1.bn3.weight
backbone.layer1.1.bn3.bias
backbone.layer1.1.bn3.running_mean
backbone.layer1.1.bn3.running_var
backbone.layer1.2.conv1.weight
backbone.layer1.2.bn1.weight
backbone.layer1.2.bn1.bias
backbone.layer1.2.bn1.running_mean
backbone.layer1.2.bn1.running_var
backbone.layer1.2.conv2.weight
backbone.layer1.2.bn2.weight
backbone.layer1.2.bn2.bias
backbone.layer1.2.bn2.running_mean
backbone.layer1.2.bn2.running_var
backbone.layer1.2.conv3.weight
backbone.layer1.2.bn3.weight
backbone.layer1.2.bn3.bias
backbone.layer1.2.bn3.running_mean
backbone.layer1.2.bn3.running_var
backbone.layer2.0.conv1.weight
backbone.layer2.0.bn1.weight
backbone.layer2.0.bn1.bias
backbone.layer2.0.bn1.running_mean
backbone.layer2.0.bn1.running_var
backbone.layer2.0.conv2.weight
backbone.layer2.0.bn2.weight
backbone.layer2.0.bn2.bias
backbone.layer2.0.bn2.running_mean
backbone.layer2.0.bn2.running_var
backbone.layer2.0.conv3.weight
backbone.layer2.0.bn3.weight
backbone.layer2.0.bn3.bias
backbone.layer2.0.bn3.running_mean
backbone.layer2.0.bn3.running_var
backbone.layer2.0.downsample.1.weight
backbone.layer2.0.downsample.2.weight
backbone.layer2.0.downsample.2.bias
backbone.layer2.0.downsample.2.running_mean
backbone.layer2.0.downsample.2.running_var
backbone.layer2.1.conv1.weight
backbone.layer2.1.bn1.weight
backbone.layer2.1.bn1.bias
backbone.layer2.1.bn1.running_mean
backbone.layer2.1.bn1.running_var
backbone.layer2.1.conv2.weight
backbone.layer2.1.bn2.weight
backbone.layer2.1.bn2.bias
backbone.layer2.1.bn2.running_mean
backbone.layer2.1.bn2.running_var
backbone.layer2.1.conv3.weight
backbone.layer2.1.bn3.weight
backbone.layer2.1.bn3.bias
backbone.layer2.1.bn3.running_mean
backbone.layer2.1.bn3.running_var
backbone.layer2.2.conv1.weight
backbone.layer2.2.bn1.weight
backbone.layer2.2.bn1.bias
backbone.layer2.2.bn1.running_mean
backbone.layer2.2.bn1.running_var
backbone.layer2.2.conv2.weight
backbone.layer2.2.bn2.weight
backbone.layer2.2.bn2.bias
backbone.layer2.2.bn2.running_mean
backbone.layer2.2.bn2.running_var
backbone.layer2.2.conv3.weight
backbone.layer2.2.bn3.weight
backbone.layer2.2.bn3.bias
backbone.layer2.2.bn3.running_mean
backbone.layer2.2.bn3.running_var
backbone.layer2.3.conv1.weight
backbone.layer2.3.bn1.weight
backbone.layer2.3.bn1.bias
backbone.layer2.3.bn1.running_mean
backbone.layer2.3.bn1.running_var
backbone.layer2.3.conv2.weight
backbone.layer2.3.bn2.weight
backbone.layer2.3.bn2.bias
backbone.layer2.3.bn2.running_mean
backbone.layer2.3.bn2.running_var
backbone.layer2.3.conv3.weight
backbone.layer2.3.bn3.weight
backbone.layer2.3.bn3.bias
backbone.layer2.3.bn3.running_mean
backbone.layer2.3.bn3.running_var
backbone.layer3.0.conv1.weight
backbone.layer3.0.bn1.weight
backbone.layer3.0.bn1.bias
backbone.layer3.0.bn1.running_mean
backbone.layer3.0.bn1.running_var
backbone.layer3.0.conv2.weight
backbone.layer3.0.bn2.weight
backbone.layer3.0.bn2.bias
backbone.layer3.0.bn2.running_mean
backbone.layer3.0.bn2.running_var
backbone.layer3.0.conv3.weight
backbone.layer3.0.bn3.weight
backbone.layer3.0.bn3.bias
backbone.layer3.0.bn3.running_mean
backbone.layer3.0.bn3.running_var
backbone.layer3.0.downsample.1.weight
backbone.layer3.0.downsample.2.weight
backbone.layer3.0.downsample.2.bias
backbone.layer3.0.downsample.2.running_mean
backbone.layer3.0.downsample.2.running_var
backbone.layer3.1.conv1.weight
backbone.layer3.1.bn1.weight
backbone.layer3.1.bn1.bias
backbone.layer3.1.bn1.running_mean
backbone.layer3.1.bn1.running_var
backbone.layer3.1.conv2.weight
backbone.layer3.1.bn2.weight
backbone.layer3.1.bn2.bias
backbone.layer3.1.bn2.running_mean
backbone.layer3.1.bn2.running_var
backbone.layer3.1.conv3.weight
backbone.layer3.1.bn3.weight
backbone.layer3.1.bn3.bias
backbone.layer3.1.bn3.running_mean
backbone.layer3.1.bn3.running_var
backbone.layer3.2.conv1.weight
backbone.layer3.2.bn1.weight
backbone.layer3.2.bn1.bias
backbone.layer3.2.bn1.running_mean
backbone.layer3.2.bn1.running_var
backbone.layer3.2.conv2.weight
backbone.layer3.2.bn2.weight
backbone.layer3.2.bn2.bias
backbone.layer3.2.bn2.running_mean
backbone.layer3.2.bn2.running_var
backbone.layer3.2.conv3.weight
backbone.layer3.2.bn3.weight
backbone.layer3.2.bn3.bias
backbone.layer3.2.bn3.running_mean
backbone.layer3.2.bn3.running_var
backbone.layer3.3.conv1.weight
backbone.layer3.3.bn1.weight
backbone.layer3.3.bn1.bias
backbone.layer3.3.bn1.running_mean
backbone.layer3.3.bn1.running_var
backbone.layer3.3.conv2.weight
backbone.layer3.3.bn2.weight
backbone.layer3.3.bn2.bias
backbone.layer3.3.bn2.running_mean
backbone.layer3.3.bn2.running_var
backbone.layer3.3.conv3.weight
backbone.layer3.3.bn3.weight
backbone.layer3.3.bn3.bias
backbone.layer3.3.bn3.running_mean
backbone.layer3.3.bn3.running_var
backbone.layer3.4.conv1.weight
backbone.layer3.4.bn1.weight
backbone.layer3.4.bn1.bias
backbone.layer3.4.bn1.running_mean
backbone.layer3.4.bn1.running_var
backbone.layer3.4.conv2.weight
backbone.layer3.4.bn2.weight
backbone.layer3.4.bn2.bias
backbone.layer3.4.bn2.running_mean
backbone.layer3.4.bn2.running_var
backbone.layer3.4.conv3.weight
backbone.layer3.4.bn3.weight
backbone.layer3.4.bn3.bias
backbone.layer3.4.bn3.running_mean
backbone.layer3.4.bn3.running_var
backbone.layer3.5.conv1.weight
backbone.layer3.5.bn1.weight
backbone.layer3.5.bn1.bias
backbone.layer3.5.bn1.running_mean
backbone.layer3.5.bn1.running_var
backbone.layer3.5.conv2.weight
backbone.layer3.5.bn2.weight
backbone.layer3.5.bn2.bias
backbone.layer3.5.bn2.running_mean
backbone.layer3.5.bn2.running_var
backbone.layer3.5.conv3.weight
backbone.layer3.5.bn3.weight
backbone.layer3.5.bn3.bias
backbone.layer3.5.bn3.running_mean
backbone.layer3.5.bn3.running_var
backbone.layer4.0.conv1.weight
backbone.layer4.0.bn1.weight
backbone.layer4.0.bn1.bias
backbone.layer4.0.bn1.running_mean
backbone.layer4.0.bn1.running_var
backbone.layer4.0.conv2.weight
backbone.layer4.0.bn2.weight
backbone.layer4.0.bn2.bias
backbone.layer4.0.bn2.running_mean
backbone.layer4.0.bn2.running_var
backbone.layer4.0.conv3.weight
backbone.layer4.0.bn3.weight
backbone.layer4.0.bn3.bias
backbone.layer4.0.bn3.running_mean
backbone.layer4.0.bn3.running_var
backbone.layer4.0.downsample.1.weight
backbone.layer4.0.downsample.2.weight
backbone.layer4.0.downsample.2.bias
backbone.layer4.0.downsample.2.running_mean
backbone.layer4.0.downsample.2.running_var
backbone.layer4.1.conv1.weight
backbone.layer4.1.bn1.weight
backbone.layer4.1.bn1.bias
backbone.layer4.1.bn1.running_mean
backbone.layer4.1.bn1.running_var
backbone.layer4.1.conv2.weight
backbone.layer4.1.bn2.weight
backbone.layer4.1.bn2.bias
backbone.layer4.1.bn2.running_mean
backbone.layer4.1.bn2.running_var
backbone.layer4.1.conv3.weight
backbone.layer4.1.bn3.weight
backbone.layer4.1.bn3.bias
backbone.layer4.1.bn3.running_mean
backbone.layer4.1.bn3.running_var
backbone.layer4.2.conv1.weight
backbone.layer4.2.bn1.weight
backbone.layer4.2.bn1.bias
backbone.layer4.2.bn1.running_mean
backbone.layer4.2.bn1.running_var
backbone.layer4.2.conv2.weight
backbone.layer4.2.bn2.weight
backbone.layer4.2.bn2.bias
backbone.layer4.2.bn2.running_mean
backbone.layer4.2.bn2.running_var
backbone.layer4.2.conv3.weight
backbone.layer4.2.bn3.weight
backbone.layer4.2.bn3.bias
backbone.layer4.2.bn3.running_mean
backbone.layer4.2.bn3.running_var
neck.lateral_convs.0.conv.weight
neck.lateral_convs.0.conv.bias
neck.lateral_convs.1.conv.weight
neck.lateral_convs.1.conv.bias
neck.lateral_convs.2.conv.weight
neck.lateral_convs.2.conv.bias
neck.fpn_convs.0.conv.weight
neck.fpn_convs.0.conv.bias
neck.fpn_convs.1.conv.weight
neck.fpn_convs.1.conv.bias
neck.fpn_convs.2.conv.weight
neck.fpn_convs.2.conv.bias
neck.fpn_convs.3.conv.weight
neck.fpn_convs.3.conv.bias
neck.fpn_convs.4.conv.weight
neck.fpn_convs.4.conv.bias
rpn_net.cls_convs.0.weight
rpn_net.cls_convs.0.bias
rpn_net.cls_convs.1.weight
rpn_net.cls_convs.1.bias
rpn_net.cls_convs.2.weight
rpn_net.cls_convs.2.bias
rpn_net.cls_convs.3.weight
rpn_net.cls_convs.3.bias
rpn_net.reg_convs.0.weight
rpn_net.reg_convs.0.bias
rpn_net.reg_convs.1.weight
rpn_net.reg_convs.1.bias
rpn_net.reg_convs.2.weight
rpn_net.reg_convs.2.bias
rpn_net.reg_convs.3.weight
rpn_net.reg_convs.3.bias
rpn_net.retina_cls.weight
rpn_net.retina_cls.bias
rpn_net.retina_reg.weight
rpn_net.retina_reg.bias"""

def main(weight_path):
    ours_names = ours_names_.split("\n")
    # weight_path = "weights/RetinaNet_DOTA_2x_20200915_DOTA_702000model.pk"
    rule_path = weight_path + ".txt"
    data = pk.load(open(weight_path, "rb"))
    keys = list(data.keys())
    replaces = [
        ['resnet50_v1d', '0_resnet50_v1d'],
        ['BatchNorm/gamma', 'BatchNorm/0_gamma'], 
        ['BatchNorm/beta', 'BatchNorm/1_beta'], 
        ['BatchNorm/moving_mean', 'BatchNorm/2_moving_mean'], 
        ['BatchNorm/moving_variance', 'BatchNorm/3_moving_variance'], 
        ['BatchNorm', 'z_BatchNorm'],
        ['/biases', '/z_biases'], 
    ]
    keys_ = []
    for i in range(len(keys)):
        k = keys[i]
        if (k.endswith("/ExponentialMovingAverage") or k.endswith("/Momentum")) or k == "global_step":
            continue
        for rep in replaces:
            k = k.replace(rep[0], rep[1])
        keys_.append([k, keys[i]])
    keys_.sort(key=lambda a : a[0])
    pairs = []
    for i in range(len(keys_)):
        if (i < len(ours_names)):
            print(keys_[i][1], " "*(80-len(keys_[i][1])), ours_names[i])
            pairs.append([keys_[i][1], ours_names[i]])
        # print(keys_[i][1])
    import json
    json.dump(pairs, open(weight_path + '_pairs.json', "w"))
    out_data = {}
    for p in pairs:
        out_data[p[1]] = data[p[0]]
    pk.dump(out_data, open(weight_path+'_jt.pk', "wb"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str)
    args = parser.parse_args()
    main(args.infile)