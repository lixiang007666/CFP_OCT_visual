import cv2
import numpy as np

# 读取金标准和模型预测图像
img_gt = cv2.imread("/home/lixiang/CFP_OCT_visual/vessel/all_groundTruths.png", cv2.IMREAD_GRAYSCALE)
img_pred = cv2.imread("/home/lixiang/CFP_OCT_visual/vessel/all_predictions.png", cv2.IMREAD_GRAYSCALE)

# 将像素值为255的预测结果二值化为1，其他为0
_, img_pred = cv2.threshold(img_pred, 127, 1, cv2.THRESH_BINARY)

# 计算True Positive、False Negative和False Positive的掩码图像
mask_tp = np.zeros_like(img_gt)
mask_fn = np.zeros_like(img_gt)
mask_fp = np.zeros_like(img_gt)
mask_tp[(img_gt == 255) & (img_pred == 1)] = 1
mask_fn[(img_gt == 255) & (img_pred == 0)] = 1
mask_fp[(img_gt == 0) & (img_pred == 1)] = 1

# 可视化结果
img_vis_gray = np.zeros_like(img_gt)
img_vis_gray[mask_tp > 0] = 255 # True Positive
img_vis_gray[mask_fn > 0] = 128 # False Negative
img_vis_gray[mask_fp > 0] = 64 # False Positive
img_vis = cv2.cvtColor(img_vis_gray, cv2.COLOR_GRAY2BGR)

# 为不同的掩码图像分配不同的颜色
img_vis[mask_tp > 0] = (0, 255, 0) # True Positive
img_vis[mask_fn > 0] = (0, 0, 255) # False Negative
img_vis[mask_fp > 0] = (0, 140, 255) # False Positive

# 保存可视化结果
cv2.imwrite("visualization.png", img_vis)