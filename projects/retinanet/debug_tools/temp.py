
# import cv2
import jittor as jt
# img = cv2.imread("/mnt/disk/cxjyxx_me/JAD/datasets/test/DOTA/trainval/images/P0003.png")
# print(img.shape)

score_j = jt.random([10000])
bbox_j = jt.random([10000, 5]) * 200 - 100

# # Error
# mask = (score_j > 0.5) == 1
# score_j = score_j[mask]
# bbox_j = bbox_j[mask]

# mask = ((bbox_j[:, 4] < 90) + (bbox_j[:, 4] > -90)) == 2
# score_j = score_j[mask]
# bbox_j = bbox_j[mask]
# print(score_j.shape)


# Correct
mask = ((score_j>0.5)+(bbox_j[:, 4] < 90) + (bbox_j[:, 4] > -90))==3
bbox_j = bbox_j[mask,:]
score_j = score_j[mask]
print(score_j.shape)
