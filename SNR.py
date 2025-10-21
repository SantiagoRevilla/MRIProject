import os
import nibabel as nib
from scipy.ndimage import rotate
import cv2
import numpy as np
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

# **FUNCIONES**

def show(img, title):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def segment_brain(img_slice, blur_ksize=(11,11)):
    blurred = cv2.GaussianBlur(img_slice, blur_ksize, 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

patch_size = 10
def find_patch_max_std(img, brain_mask, patch_size, dilation_size=2):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
    brain_mask_dilated = cv2.dilate(brain_mask, kernel)
    max_std = 0
    patch_coords = None
    for y in range(0, img.shape[0] - patch_size):
        for x in range(0, img.shape[1] - patch_size):
            patch_mask = brain_mask_dilated[y:y+patch_size, x:x+patch_size]
            patch_img = img[y:y+patch_size, x:x+patch_size]

            if np.all(patch_mask == 0) and np.all(patch_img > 0):
                std_val = np.std(patch_img)
                if std_val > max_std:
                    max_std = std_val
                    patch_coords = (x, y)
    return patch_coords, max_std, brain_mask_dilated

def find_patch_min_mean(img, brain_mask, patch_size):
    min_mean = float('inf')
    patch_coords = None
    for y in range(0, img.shape[0] - patch_size):
        for x in range(0, img.shape[1] - patch_size):
            patch_mask = brain_mask[y:y+patch_size, x:x+patch_size]
            patch_img = img[y:y+patch_size, x:x+patch_size]
            if np.all(patch_mask > 0):
                mean_val = np.mean(patch_img)
                if mean_val < min_mean:
                    min_mean = mean_val
                    patch_coords = (x, y)
    return patch_coords, min_mean

#**LISA 0002 Sagital RUIDO: 1**

# Cargar archivo .nii
image = nib.load('/content/drive/MyDrive/Lisa/LISA_0002_LF_sag.nii')
data = image.get_fdata()

# Elegir corte y rotar
i=14
img = rotate(data[i], 90, reshape=False)
show(img, f'Original Rotated: {i}')
print(img.shape)

# Normalizar imagen
img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
img_u8 = img_norm.astype(np.uint8)
show(img_u8, 'Imagen Normalizada')

brain_mask = segment_brain(img_u8)
show(brain_mask, 'Brain Mask')

pos_patch_ext, std_patch, brain_mask_dilated = find_patch_max_std(img_u8, brain_mask, patch_size)

# Resultado
img_with_patch_ext = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
if pos_patch_ext:
    x, y = pos_patch_ext
    cv2.rectangle(img_with_patch_ext, (x, y), (x+patch_size, y+patch_size), (0, 255, 0), 2)
    print(f'Valor de x: {x}')
    print(f'Valor de y: {y}')
    print(f'Desviación estándar del patch: {std_patch}')

plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(img_with_patch_ext, cv2.COLOR_BGR2RGB))
plt.title('Zona 10x10 en Background Cercano')
plt.axis('on')
plt.show()

# Visualizar máscara dilatada
plt.figure(figsize=(6, 6))
plt.imshow(brain_mask_dilated, cmap='gray')
plt.title('Máscara dilatada')
plt.axis('on')
plt.show()

pos_patch_int, mean_patch = find_patch_min_mean(img_u8, brain_mask, patch_size)

# Resultado
img_with_patch_int = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
if pos_patch_int:
    x, y = pos_patch_int
    cv2.rectangle(img_with_patch_int, (x, y), (x+patch_size, y+patch_size), (255, 0, 0), 2)
    print(f'Valor de x: {x}')
    print(f'Valor de y: {y}')
    print(f'Media del patch interno: {mean_patch:.2f}')

plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(img_with_patch_int, cv2.COLOR_BGR2RGB))
plt.title('Patch interno con media más baja')
plt.axis('on')
plt.show()

SNR = mean_patch / std_patch
print(f'SNR: {SNR}')

# **LISA 0037 Sagital RUIDO: 2**

# Cargar archivo .nii
image1 = nib.load('/content/drive/MyDrive/Lisa/LISA_0037_LF_sag.nii')
data1 = image1.get_fdata()

# Elegir corte y rotar
j=16
img1 = rotate(data1[j], 90, reshape=False)
show(img1, f'Original Rotated: {j}')
print(img1.shape)

# Normalizar imagen
img1_norm = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX)
img1_u8 = img1_norm.astype(np.uint8)
show(img1_u8, 'Imagen Normalizada')

brain_mask1 = segment_brain(img1_u8)
show(brain_mask1, 'Brain Mask')

pos_patch_ext1, std_patch1, brain_mask_dilated1 = find_patch_max_std(img1_u8, brain_mask1, patch_size)

# Resultado
img_with_patch_ext1 = cv2.cvtColor(img1_u8, cv2.COLOR_GRAY2BGR)
if pos_patch_ext1:
    x, y = pos_patch_ext1
    cv2.rectangle(img_with_patch_ext1, (x, y), (x+patch_size, y+patch_size), (0, 255, 0), 2)
    print(f'Valor de x: {x}')
    print(f'Valor de y: {y}')
    print(f'Desviación estándar del patch: {std_patch1}')

plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(img_with_patch_ext1, cv2.COLOR_BGR2RGB))
plt.title('Zona 10x10 en Background Cercano')
plt.axis('on')
plt.show()

# Visualizar máscara dilatada
plt.figure(figsize=(6, 6))
plt.imshow(brain_mask_dilated1, cmap='gray')
plt.title('Máscara dilatada')
plt.axis('on')
plt.show()

pos_patch_int1, mean_patch1 = find_patch_min_mean(img1_u8, brain_mask1, patch_size)

# Resultado
img_with_patch_int1 = cv2.cvtColor(img1_u8, cv2.COLOR_GRAY2BGR)
if pos_patch_int1:
    x, y = pos_patch_int1
    cv2.rectangle(img_with_patch_int1, (x, y), (x+patch_size, y+patch_size), (255, 0, 0), 2)
    print(f'Valor de x: {x}')
    print(f'Valor de y: {y}')
    print(f'Media del patch interno: {mean_patch1:.2f}')

plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(img_with_patch_int1, cv2.COLOR_BGR2RGB))
plt.title('Patch interno con media más baja')
plt.axis('on')
plt.show()

SNR1 = mean_patch1 / std_patch1
print(f'SNR: {SNR1}')

# **LISA 1017 Sagital RUIDO: 2**

# Cargar archivo .nii
image2 = nib.load('/content/drive/MyDrive/Lisa/LISA_1017_LF_sag.nii')
data2 = image2.get_fdata()

# Elegir corte y rotar
k=12
img2 = rotate(data2[k], 90, reshape=False)
show(img2, f'Original Rotated: {k}')
print(img2.shape)

# Normalizar imagen
img2_norm = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX)
img2_u8 = img2_norm.astype(np.uint8)
show(img2_u8, 'Imagen Normalizada')

brain_mask2 = segment_brain(img2_u8)
show(brain_mask2, 'Brain Mask')

pos_patch_ext2, std_patch2, brain_mask_dilated2 = find_patch_max_std(img2_u8, brain_mask2, patch_size)

# Resultado
img_with_patch_ext2 = cv2.cvtColor(img2_u8, cv2.COLOR_GRAY2BGR)
if pos_patch_ext2:
    x, y = pos_patch_ext2
    cv2.rectangle(img_with_patch_ext2, (x, y), (x+patch_size, y+patch_size), (0, 255, 0), 2)
    print(f'Valor de x: {x}')
    print(f'Valor de y: {y}')
    print(f'Desviación estándar del patch: {std_patch2}')

plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(img_with_patch_ext2, cv2.COLOR_BGR2RGB))
plt.title('Zona 10x10 en Background Cercano')
plt.axis('on')
plt.show()

# Visualizar máscara dilatada
plt.figure(figsize=(6, 6))
plt.imshow(brain_mask_dilated2, cmap='gray')
plt.title('Máscara dilatada')
plt.axis('on')
plt.show()

pos_patch_int2, mean_patch2 = find_patch_min_mean(img2_u8, brain_mask2, patch_size)

# Resultado
img_with_patch_int2 = cv2.cvtColor(img2_u8, cv2.COLOR_GRAY2BGR)
if pos_patch_int2:
    x, y = pos_patch_int2
    cv2.rectangle(img_with_patch_int2, (x, y), (x+patch_size, y+patch_size), (255, 0, 0), 2)
    print(f'Valor de x: {x}')
    print(f'Valor de y: {y}')
    print(f'Media del patch interno: {mean_patch2:.2f}')

plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(img_with_patch_int2, cv2.COLOR_BGR2RGB))
plt.title('Patch interno con media más baja')
plt.axis('on')
plt.show()

SNR2 = mean_patch2 / std_patch2
print(f'SNR: {SNR2}')