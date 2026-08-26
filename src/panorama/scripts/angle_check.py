import numpy as np
from math import atan2, acos, sqrt

# 输入参数
u_new = 10    # 新像素u'
v_new = 255    # 新像素v'
W = 640       # 原始图像宽度
H = 480       # 原始图像高度
f_u = 429.733259611991     # 原始焦距u
f_v = 429.841866168175     # 原始焦距v
c_u = 321.365347200167      # 原始主点u
c_v = 245.160847403009      # 原始主点v

# 步骤1: 映射到原始像素坐标
u_orig = W - 1 - v_new
v_orig = u_new

# 步骤2: 计算原始bearing vector (归一化)
x_orig = (u_orig - c_u) / f_u
y_orig = (v_orig - c_v) / f_v
b_orig = np.array([x_orig, y_orig, 1.0])
norm_orig = np.linalg.norm(b_orig)
b_orig_normalized = b_orig / norm_orig

# 步骤3: 应用旋转
R_c = np.array([[0, -1, 0],
                [1, 0, 0],
                [0, 0, 1]])
b_new = R_c @ b_orig_normalized  # 矩阵乘法

print(u_orig)
print(v_orig)

# # 步骤4: 计算角度
# theta = acos(b_new[2])  # 俯仰角, 弧度
# phi = atan2(b_new[1], b_new[0])  # 方位角, 弧度

# # 转换为度数
# theta_deg = np.degrees(theta)
# phi_deg = np.degrees(phi)
print(f"Bearing vector in new coord: {b_new}")
# print(f"Pitch angle (θ): {theta_deg:.1f} degrees")
# print(f"Azimuth angle (φ): {phi_deg:.1f} degrees")