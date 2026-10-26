import numpy as np
import cv2
import matplotlib.pyplot as plt

A = cv2.imread('pic.jpg', cv2.IMREAD_GRAYSCALE).astype(float)
U, S, Vt = np.linalg.svd(A, full_matrices=False)
print(A.shape, S.shape, len(S))

'''
ks = [1, 5, 10, 20, 30, 50, 70, 100, 200]
total = np.sum(S**2)
captured = np.cumsum(S**2) / total
for k in ks:
    Ak = (U[:, :k] * S[:k]) @ Vt[:k, :]
    err_svd = np.sqrt(1 - captured[k-1])
    storage = k * (U.shape[0] + Vt.shape[1] + 1) / (A.shape[0] * A.shape[1])
    print(f"k={k:>3}  svd={err_svd:.10f}  storage={storage:.10f}")
''' #检验svd压缩的误差和存储量

ks_all = np.arange(1, len(S) + 1)
err_all = np.sqrt(1 - np.cumsum(S**2) / np.sum(S**2))
ratio_all = ks_all * (U.shape[0] + Vt.shape[1] + 1) / (A.shape[0] * A.shape[1])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

ax1.plot(ks_all, err_all * 100)
ax1.set_xlabel("rank k")
ax1.set_ylabel("relative error (%)")
ax1.set_title("Error vs rank")
ax1.grid(alpha=0.3)

ax2.plot(ratio_all * 100, err_all * 100)
ax2.set_xlabel("storage (% of original)")
ax2.set_ylabel("relative error (%)")
ax2.set_title("Error vs storage")
ax2.grid(alpha=0.3)

ax2.axvline(100, color='gray', linestyle='--')

plt.tight_layout()
plt.savefig('error_curve.png', dpi=150)
plt.close(fig)
