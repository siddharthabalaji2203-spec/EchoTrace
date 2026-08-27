
import cv2
import matplotlib.pyplot as plt
import numpy as np
from src.preprocessing.physics import (
    load_grayscale,
    apply_range_compensation,
)

image = load_grayscale(r"C:\Users\Siddhartha\Desktop\EchoTrace\data\KLSG\archive\test\images\000109_jpg.rf.a22cc4e5913e2a401835bd836e4bc999.jpg")

corrected = apply_range_compensation(
    image,
    strength=0.25
)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original SSS")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(corrected, cmap="gray")
plt.title("Range Compensated")
plt.axis("off")

plt.tight_layout()
plt.savefig("physics_comparison.png", dpi=200)
plt.close()

print("Comparison saved: physics_comparison.png")

print("Original mean:", np.mean(image))
print("Corrected mean:", np.mean(corrected))
print("Original left mean:", np.mean(image[:, :image.shape[1] // 4]))
print("Original right mean:", np.mean(image[:, -image.shape[1] // 4:]))
print("Corrected left mean:", np.mean(corrected[:, :corrected.shape[1] // 4]))
print("Corrected right mean:", np.mean(corrected[:, -corrected.shape[1] // 4:]))
