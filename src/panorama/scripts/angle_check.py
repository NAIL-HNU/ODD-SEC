import numpy as np
from math import atan2, acos, sqrt

u_new = 10
v_new = 255
W = 640
H = 480
f_u = 429.733259611991
f_v = 429.841866168175
c_u = 321.365347200167
c_v = 245.160847403009

u_orig = W - 1 - v_new
v_orig = u_new

x_orig = (u_orig - c_u) / f_u
y_orig = (v_orig - c_v) / f_v
b_orig = np.array([x_orig, y_orig, 1.0])
norm_orig = np.linalg.norm(b_orig)
b_orig_normalized = b_orig / norm_orig

R_c = np.array([[0, -1, 0],
                [1, 0, 0],
                [0, 0, 1]])
b_new = R_c @ b_orig_normalized

print(u_orig)
print(v_orig)

print(f"Bearing vector in new coord: {b_new}")
