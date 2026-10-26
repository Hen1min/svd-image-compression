import numpy as np
import cv2
import matplotlib.pyplot as plt

A = cv2.imread('pic.jpg', cv2.IMREAD_GRAYSCALE).astype(float)
U, S, V = np.linalg.svd(A, full_matrices=False)
print("前十个奇异值：", S[:10])

ks = [10, 30, 50, 100]
fig, axes = plt.subplots(1, 1 + len(ks), figsize=(15, 5))
axes[0].imshow(A, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

for ax, k in zip(axes[1:], ks):
    Ak = (U[:, :k] * S[:k]) @ V[:k, :]
    Ak = np.clip(Ak, 0, 255).astype(np.uint8) # Clip values to valid range and convert to uint8
    ax.imshow(Ak, cmap='gray')
    ax.set_title(f'Rank-{k} Approximation')
    ax.axis('off')

plt.tight_layout()
plt.savefig('rank_comparison.png', dpi=150)
plt.close(fig)