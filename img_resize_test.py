import cv2

im = cv2.imread("/home/dio/VSST/xsq/Dataset/DOTA-v1.5/obb_val/images/P1809__1__2472___824.png")
assert im is not None
h0,w0 = 1024,1024
r = 640 / max(h0,w0)
im = cv2.resize(im, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_AREA if r > 1 else cv2.INTER_LINEAR)
cv2.imwrite("resize_1.png",im)
