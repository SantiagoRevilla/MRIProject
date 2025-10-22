import nibabel as nib
from scipy.ndimage import rotate
import numpy as np
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

def show(img, title):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def fft2c(img):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))

def ifft2c(kspace):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace)))

# Cargar archivo .nii
image = nib.load('/content/drive/MyDrive/Lisa/LISA_0057_LF_sag.nii')
data = image.get_fdata()

# Elegir corte y rotar
i=16
img = rotate(data[i], 90, reshape=False)
show(img, f'Original Rotated: {i}')
print(img.shape)

kdata = fft2c(img)
im_original = img.copy()
show(np.abs(kdata), 'k-space magnitude')

relative_RF_frequency = 0.3
S = img.shape
rfint = np.exp(1j * 2 * np.pi * np.arange(S[0]) * relative_RF_frequency / 2) * np.max(np.abs(kdata)) / S[0] / 3
data_rfint = (np.random.randn(S[0], 1)) * rfint  # amplitude modulation
im_rfint = ifft2c(kdata + data_rfint)

plt.subplot(1, 2, 1)
plt.imshow(np.log(np.abs(kdata + data_rfint) + 1e1), cmap='gray', aspect='equal')
plt.axis('off')
plt.title('k-space with RF int. (log)')
plt.subplot(1, 2, 2)
plt.imshow(np.abs(im_rfint), cmap='gray', aspect='equal', vmin=0, vmax=np.max(np.abs(im_original)))
plt.axis('off')
plt.title('RF Interference Artifact')
plt.show()

# --- Escalar suavemente la banda del zipper ---
zipper_band = im_rfint_full * mask  # extraemos solo la banda
zipper_band_scaled = zipper_band / np.max(zipper_band) * np.max(np.abs(im_original))

# Creamos la imagen final mezclando la banda suavizada con el resto de la imagen
im_rfint_suave = np.abs(im_original) * (1 - mask) + zipper_band_scaled

# --- Visualización en una 4ta imagen ---
plt.figure(figsize=(12,4))

plt.subplot(1,4,1)
plt.imshow(np.log(np.abs(kdata_rf) + 1e1), cmap='gray', aspect='equal')
plt.axis('off')
plt.title('k-space with RF int. (log)')

plt.subplot(1,4,2)
plt.imshow(np.abs(im_rfint_full), cmap='gray', aspect='equal',
           vmin=0, vmax=np.max(np.abs(im_original)))
plt.axis('off')
plt.title('RF interference (full)')

plt.subplot(1,4,3)
plt.imshow(np.abs(im_original), cmap='gray', aspect='equal')
plt.axis('off')
plt.title('Imagen original')

plt.subplot(1,4,4)
plt.imshow(im_rfint_suave, cmap='gray', aspect='equal',
           vmin=0, vmax=np.max(np.abs(im_original)))
plt.axis('off')
plt.title('RF interference (banda suavizada)')
plt.show()


# --- Suavizar la intensidad del zipper completo ---
alpha = 0.3  # 0 = solo imagen original, 1 = solo interferencia completa
rf_mag = np.abs(im_rfint_full)
# Escalamos la interferencia para que no supere el máximo de la imagen
rf_mag_scaled = rf_mag / (np.max(rf_mag) + 1e-12) * np.max(np.abs(im_original))
# Mezcla suavizada
im_rfint_smoothed = np.abs(im_original) * (1 - alpha) + rf_mag_scaled * alpha

# --- Aplicar máscara vertical sobre la imagen suavizada ---
mask = np.zeros_like(im_rfint_smoothed)
band_center = 52  # posición central de la banda
band_width  = 10  # ancho de la banda
x_start = max(band_center - band_width // 2, 0)
x_end   = min(band_center + band_width // 2, S[1])
mask[:, x_start:x_end] = 1

# Combinamos la imagen original y la banda vertical con interferencia suavizada
im_rfint_masked = np.abs(im_original) * (1 - mask) + im_rfint_smoothed * mask

# --- Visualización ---
plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.imshow(np.log(np.abs(kdata_rf) + 1e1), cmap='gray', aspect='equal')
plt.axis('off')
plt.title('k-space with RF int. (log)')

plt.subplot(1, 4, 2)
plt.imshow(np.abs(im_rfint_full), cmap='gray', aspect='equal',
           vmin=0, vmax=np.max(np.abs(im_original)))
plt.axis('off')
plt.title('RF interference (full image)')

plt.subplot(1, 4, 3)
plt.imshow(im_rfint_smoothed, cmap='gray', aspect='equal',
           vmin=0, vmax=np.max(np.abs(im_original)))
plt.axis('off')
plt.title('Zipper suavizado (full image)')

plt.subplot(1, 4, 4)
plt.imshow(im_rfint_masked, cmap='gray', aspect='equal',
           vmin=0, vmax=np.max(np.abs(im_original)))
plt.axis('off')
plt.title(f'Zipper suavizado + banda vertical ({band_center}, {band_width})')

plt.show()
