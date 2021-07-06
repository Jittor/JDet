import pickle as pk
data = pk.load(open("weights/RetinaNet_DOTA_2x_20200915_DOTA_702000model.pk", "rb"))
ks = list(data.keys())
ks.sort()
for k in ks:
    if (k.startswith("resnet50_v1d/C1")):
        print(k)