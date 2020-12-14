import cv2
import numpy as np

w = 256
h = 256
th = 1

# create 3 separate BGRA images as our "layers"
layer1 = np.zeros((h, w, 4), np.uint8)
layer2 = np.zeros((h, w, 4), np.uint8)
layer3 = np.zeros((h, w, 4), np.uint8)

# draw a red circle on the first "layer",
# a green rectangle on the second "layer",
# a blue line on the third "layer"
red_color = (0, 0, 255, 255)
green_color = (0, 255, 0, 255)
blue_color = (255, 0, 0, 255)

cv2.ellipse(
    layer1,
    (w // 2, h // 2),
    (w // 3, h // 3),
    0,
    0,
    360,
    red_color,
    th,
    lineType=cv2.LINE_8,
)

cv2.rectangle(layer2, (0, 0), (w-1, h-1), green_color, th, lineType=cv2.LINE_4)
cv2.line(layer3, (0, 0), (w-1, h-1), blue_color, th, lineType=cv2.LINE_AA)
cv2.line(layer3, (0, h-1), (w-1, 0), blue_color, th, lineType=cv2.LINE_AA)

res = layer1[:]  # copy the first layer into the resulting image

# copy only the pixels we were drawing on from the 2nd and 3rd layers
# (if you don't do this, the black background will also be copied)
cnd = layer2[:, :, 3] > 0
res[cnd] = layer2[cnd]
cnd = layer3[:, :, 3] > 0
res[cnd] = layer3[cnd]

res2 = cv2.resize(res, (300, 400), interpolation=cv2.INTER_CUBIC)

cv2.imwrite("out.png", res2)


low1 = cv2.pyrDown(res)
low2 = cv2.pyrDown(low1)
high1 = cv2.pyrUp(res)
high2 = cv2.pyrUp(high1)
cv2.imwrite("out_high1.png", high1)
cv2.imwrite("out_low1.png", low1)
cv2.imwrite("out_high2.png", high2)
cv2.imwrite("out_low2.png", low2)
