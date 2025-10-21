## SENSE

import nibabel as nib
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from google.colab import drive
drive.mount('/content/drive')

# FFT centrada conveniencia
def fft2c(x):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))

def ifft2c(X):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(X)))

def show(img, title="", cmap="gray", vmin=None, vmax=None):
    plt.figure(figsize=(5,5))
    if np.iscomplexobj(img):
        img_to_show = np.log1p(np.abs(img))
        plt.imshow(img_to_show, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.axis("off")
    plt.show()

# Módulo 1 – Lectura de datos NIfTI y normalización

NIFTI_PATH = "/content/drive/MyDrive/5to Semestre /LISA_0007_LF_sag (1).nii"  # cambia si hace falta

img_nii = nib.load(NIFTI_PATH)
vol = img_nii.get_fdata()

slice_idx = 20  # cambia si quieres otra rebanada
m = vol[slice_idx].astype(np.float32)
# opcional: rotar si la orientación lo requiere
m = ndi.rotate(m, 90, reshape=False)

# normalizar a 0..1 para estabilidad
m = m - m.min()
m = m / (m.max() + 1e-12)

print("Tamaño de la slice:", m.shape)
show(m, f"Imagen (slice {slice_idx})")

# Módulo 2 - Mapas de sensibilidad

# Módulo 2 - mapas de sensibilidad (S_i) sintéticos suaves
# Ecuación: S_i(x,y) = magnitud_suave(x,y) * exp(j * fase(x,y))
# Generamos N coils distribuidos alrededor del centro

def make_sensitivities(shape, num_coils=8, spread=0.6, add_phase=True):
    nx, ny = shape
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    S = np.zeros((nx, ny, num_coils), dtype=np.complex64)
    for k in range(num_coils):
        ang = 2*np.pi * k / num_coils
        cx = 0.6 * np.cos(ang)
        cy = 0.6 * np.sin(ang)
        mag = np.exp(-((X-cx)**2 + (Y-cy)**2)/spread)  # perfil gaussiano
        if add_phase:
            phase = np.exp(1j * (np.pi * (X*cx + Y*cy)))  # fase suave dependiente de posición
            S[..., k] = mag * phase
        else:
            S[..., k] = mag
    # normalizar por su máximo (opcional)
    S = S / (np.max(np.abs(S), axis=(0,1), keepdims=True) + 1e-12)
    return S

num_coils = 8
S = make_sensitivities(m.shape, num_coils=num_coils, spread=0.7, add_phase=True)

# Mostrar mapas de sensibilidad (magnitud)
plt.figure(figsize=(12,3))
for i in range(num_coils):
    plt.subplot(1, num_coils, i+1)
    plt.imshow(np.abs(S[...,i]), cmap='viridis')
    plt.title(f"S_{i+1}")
    plt.axis('off')
plt.suptitle("Mapas de sensibilidad (magnitud)")
plt.show()

# Módulo 3 - señales por bobina en imagen y  k-space


b_imgs = S * m[..., None]            # shape (nx,ny,ncoils)
B = np.stack([fft2c(b_imgs[..., i]) for i in range(num_coils)], axis=-1)  # k-space por coil

print("b_imgs shape:", b_imgs.shape)
print("B (k-space) shape:", B.shape)

# Mostrar algunas imágenes por bobina (imagen espacial)
plt.figure(figsize=(12,3))
for i in range(num_coils):
    plt.subplot(1, num_coils, i+1)
    plt.imshow(np.abs(b_imgs[...,i]), cmap='gray')
    plt.title(f"b_{i+1} (imagen)")
    plt.axis('off')
plt.suptitle("Imágenes por bobina (espacio imagen)")
plt.show()

# Mostrar k-space (log-magnitud) por bobina - útil para inspección
plt.figure(figsize=(12,3))
for i in range(num_coils):
    plt.subplot(1, num_coils, i+1)
    plt.imshow(np.log1p(np.abs(B[...,i])), cmap='gray')
    plt.title(f"K_{i+1} (log|K|)")
    plt.axis('off')
plt.suptitle("K-space (log magnitud) por bobina")
plt.show()


# Módulo 4 – Ruido

# Módulo 4 - añadir ruido en k-space y reconstruir
# Bn = B + noise (ruido complejo en k-space)
sigma_rel = 0.007             # nivel relativo (ajusta)
sigma = sigma_rel * np.max(np.abs(B))
noise = sigma * (np.random.randn(*B.shape) + 1j*np.random.randn(*B.shape))
Bn = B + noise

# Mostrar k-space ruidoso (log)
plt.figure(figsize=(12,3))
for i in range(num_coils):
    plt.subplot(1, num_coils, i+1)
    plt.imshow(np.log1p(np.abs(Bn[...,i])), cmap='gray')
    plt.title(f"K_{i+1} noisy")
    plt.axis('off')
plt.suptitle("K-space con ruido (log)")
plt.show()

# Reconstruir por bobina (imagen)
bn_imgs = np.stack([ifft2c(Bn[..., i]) for i in range(num_coils)], axis=-1)

# RSS reconstrucción
m_RSS = np.sqrt(np.sum(np.abs(bn_imgs)**2, axis=-1))

# Mostrar imágenes por bobina
plt.figure(figsize=(12,4))
for i in range(num_coils):
    plt.subplot(2, num_coils//2, i+1)
    plt.imshow(np.abs(bn_imgs[...,i]), cmap='gray')
    plt.title(f"bobina {i+1}")
    plt.axis('off')
plt.suptitle("Imágenes reconstruidas por bobina")
plt.show()

# Mostrar reconstrucción RSS aparte
plt.figure(figsize=(5,5))
plt.imshow(m_RSS, cmap='gray')
plt.title("Reconstrucción RSS")
plt.axis('off')
plt.show()

# Módulo 5 – Estimación de sensibilidad

#Estimar mapa de sensibilidad desde las imágenes por bobina
# 1) construir m_body proxy: podemos usar una versión suavizada de m_RSS
Ib = ndi.gaussian_filter(m_RSS, sigma=0.015)    # "cuerpo" de baja resolución / menos ruido

show(Ib, "Imagen cuerpo (proxy) - suavizada")

# 2) máscara del objeto (umbral + limpieza) para evitar divisiones por fondo
th = threshold_otsu(Ib)
mask = Ib > (0.15 * Ib.max())   # puedes ajustar 0.15 como en el PDF
# limpieza morfológica
mask = ndi.binary_opening(mask, structure=np.ones((3,3)))
mask = ndi.binary_closing(mask, structure=np.ones((5,5)))
mask = ndi.binary_fill_holes(mask)

show(mask.astype(float), "Máscara estimada (objeto)")

# 3) estimación inicial sin suavizar
eps = 1e-6
S_est_raw = bn_imgs / (Ib[..., None] + eps)   # (nx,ny,ncoils)

# 4) aplicar suavizado espacial a cada mapa de sensibilidad (real e imag separados)
S_est = np.zeros_like(S_est_raw)
for i in range(num_coils):
    re = ndi.gaussian_filter(np.real(S_est_raw[...,i]) * mask, sigma=3)
    im = ndi.gaussian_filter(np.imag(S_est_raw[...,i]) * mask, sigma=3)
    # re-im normalization opcional (mantener magnitud en rango)
    S_est[...,i] = (re + 1j*im)

# Normalizar para evitar amplitude arbitraria
S_est = S_est / (np.max(np.abs(S_est), axis=(0,1), keepdims=True) + 1e-12)

# Mostrar mapas estimados (magnitud)
plt.figure(figsize=(12,3))
for i in range(num_coils):
    plt.subplot(1, num_coils, i+1)
    plt.imshow(np.abs(S_est[...,i]), cmap='viridis')
    plt.title(f"S_est {i+1}")
    plt.axis('off')
plt.suptitle("Mapas de sensibilidad estimados (magnitud)")
plt.show()


# Módulo 6 – Reconstrucción RSS

# Módulo 6 - comparaciones visuales y MSE
def mse(a,b):
    return np.mean((np.abs(a) - np.abs(b))**2)

print("MSE RSS vs original:", mse(m_RSS, m))
# mostrar comparación lado a lado
plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.imshow(m, cmap='gray'); plt.title("Imagen original (m)"); plt.axis('off')
plt.subplot(1,3,2); plt.imshow(m_RSS, cmap='gray'); plt.title("Reconstrucción RSS"); plt.axis('off')
# mostrar una reconstrucción por bobina (ej. coil 1)
plt.subplot(1,3,3); plt.imshow(np.abs(bn_imgs[...,0]), cmap='gray'); plt.title("Bobina 1 (recon)"); plt.axis('off')
plt.show()


# Módulo 7 – Submuestreo

# Aquí U es una máscara binaria en k-space (1 = muestrea, 0 = omite)

def undersampling_mask(shape, R=2, axis=0):
    """
    Genera máscara binaria para submuestreo en espacio-k.
    Por defecto omite líneas a lo largo del eje vertical (axis=0).
    """
    mask = np.zeros(shape, dtype=np.float32)
    if axis == 0:  # subsample en filas
        mask[::R, :] = 1
    else:          # subsample en columnas
        mask[:, ::R] = 1
    return mask

R = 2  # factor de aceleración
mask = undersampling_mask(m.shape, R=R, axis=0)

show(mask, f"Máscara de submuestreo (R={R})", cmap="gray")


# Módulo 8 - operadores E y E^H (SENSE)

# Módulo 8 - operadores E y E^H (SENSE)

def E(m, S, mask):
    """Aplica operador E a la imagen m (2D real/complex) usando mapas S y máscara en k-space."""
    num_coils = S.shape[-1]
    res = []
    for i in range(num_coils):
        b = S[..., i] * m
        B = fft2c(b)
        res.append(mask * B)
    return np.stack(res, axis=-1)

def EH(B, S, mask):
    """Aplica operador adjunto E^H a datos por bobina en k-space."""
    num_coils = S.shape[-1]
    acc = np.zeros(S.shape[:2], dtype=np.complex64)
    for i in range(num_coils):
        b = ifft2c(mask * B[..., i])
        acc += np.conj(S[..., i]) * b
    return acc


# Módulo 9 - aplicar submuestreo al k-space ruidoso

Bn_us = Bn * mask[..., None]

# Mostrar k-space submuestreado de una bobina
show(np.log1p(np.abs(Bn_us[...,0])), "K-space bobina 1 submuestreado (log)")


# Módulo 10 - reconstrucción aliasing (zero-filled)

bn_imgs_us = np.stack([ifft2c(Bn_us[..., i]) for i in range(num_coils)], axis=-1)
m_RSS_us = np.sqrt(np.sum(np.abs(bn_imgs_us)**2, axis=-1))

show(m_RSS_us, "Reconstrucción RSS submuestreada (aliasing)")

# Módulo 11 – Reconstrucción SENSE

def cg_sense(E, EH, B, S, mask, max_iter=20, tol=1e-6):
    """
    Resuelve (E^H E) m = E^H B con gradiente conjugado.
    """
    # b = E^H B
    b = EH(B, S, mask)
    m = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rsold = np.vdot(r, r)
    history = [np.sqrt(rsold)]

    for it in range(max_iter):
        Ap = EH(E(p, S, mask), S, mask)
        alpha = rsold / (np.vdot(p, Ap) + 1e-12)
        m = m + alpha * p
        r = r - alpha * Ap
        rsnew = np.vdot(r, r)
        history.append(np.sqrt(rsnew))
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew/rsold) * p
        rsold = rsnew
    return m, history

m_sense, hist = cg_sense(E, EH, Bn_us, S_est, mask, max_iter=30)

show(np.abs(m_sense), "Reconstrucción SENSE (CG)")
plt.plot(hist); plt.title("Convergencia CG"); plt.xlabel("iter"); plt.ylabel("||r||"); plt.show()

# Módulo 12 - Comparaciones finales

plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.imshow(m, cmap='gray'); plt.title("Original"); plt.axis('off')
plt.subplot(1,3,2); plt.imshow(m_RSS_us, cmap='gray'); plt.title("RSS submuestread (aliasing)"); plt.axis('off')
plt.subplot(1,3,3); plt.imshow(np.abs(m_sense), cmap='gray'); plt.title("Reconstrucción SENSE"); plt.axis('off')
plt.show()

# SNR


import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.morphology import disk, binary_opening, binary_closing
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import rotate

# --- Función show para imágenes grandes ---
def show(image, title):
    plt.figure(figsize=(6,6))
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# --- Funciones de segmentación y parche ---
def segment_brain(img_slice, upscale_factor=2, threshold_relax=0.95):
    if upscale_factor > 1:
        img_slice = cv2.resize(img_slice, None, fx=upscale_factor, fy=upscale_factor,
                               interpolation=cv2.INTER_CUBIC)
    float_img = img_slice.astype(np.float32)
    nonzero_pixels = float_img[float_img > 0]
    otsu_thr = threshold_otsu(nonzero_pixels)
    threshold = otsu_thr * threshold_relax
    mask = float_img > threshold
    mask = binary_closing(mask, footprint=disk(3))
    mask = binary_opening(mask, footprint=disk(2))
    mask = binary_fill_holes(mask)
    cleaned_u8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(cleaned_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_final = np.zeros_like(cleaned_u8)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask_final, [largest], -1, 255, thickness=-1)
    mask_float = mask_final.astype(np.float32) / 255.0
    mask_smooth = cv2.GaussianBlur(mask_float, (7, 7), 0)
    mask_final = (mask_smooth > 0.4).astype(np.uint8) * 255
    return mask_final

def find_patch_near_brain_border(img, brain_mask, patch_size=10):
    dilated = cv2.dilate(brain_mask, np.ones((3, 3), np.uint8), iterations=2)
    for y in range(0, img.shape[0] - patch_size):
        for x in range(0, img.shape[1] - patch_size):
            patch_mask = brain_mask[y:y+patch_size, x:x+patch_size]
            patch_img = img[y:y+patch_size, x:x+patch_size]
            near_edge = dilated[y:y+patch_size, x:x+patch_size]
            if np.all(patch_mask == 0) and np.all(patch_img > 0) and np.any(near_edge > 0):
                return (x, y)
    return None

# --- BLOQUE: estimación SNR + corrección Rician ---
i = 22
img = rotate(data[i], 90, reshape=False)
img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
img_u8 = img_norm.astype(np.uint8)

# --- Máscara de background ---
bg_thr = threshold_otsu(img_u8[img_u8>0]) * 0.6
bg_mask = (img_u8 <= bg_thr) & (img_u8 > 0)
show(bg_mask.astype(float), "Máscara de fondo (histograma/Otsu)")
sigma_est = np.std(img_u8[bg_mask])
print(f"Estimación sigma (ruido) = {sigma_est:.4f}")

# --- Máscara de señal ---
brain_mask_segmented = segment_brain(img_u8, upscale_factor=2, threshold_relax=0.95)

# Redimensionar máscara al tamaño original para que coincida con img_u8
sig_mask = cv2.resize(brain_mask_segmented.astype(np.uint8),
                      (img_u8.shape[1], img_u8.shape[0]),
                      interpolation=cv2.INTER_NEAREST).astype(bool)
show(sig_mask.astype(float), "Máscara señal")

# --- Cálculo SNR y corrección Rician ---
signal_vals = img_u8[sig_mask]
mu_signal = np.mean(signal_vals)
snr_simple = mu_signal / sigma_est
s_hat = np.sqrt(max(mu_signal**2 - 2*sigma_est**2, 0))
snr_rician_corrected = s_hat / sigma_est if sigma_est > 0 else np.nan


# --- Seleccionar un parche 10x10 dentro del cerebro ---
def find_patch_in_brain(img, brain_mask, patch_size=10):
    for y in range(0, img.shape[0] - patch_size, patch_size):
        for x in range(0, img.shape[1] - patch_size, patch_size):
            patch_mask = brain_mask[y:y+patch_size, x:x+patch_size]
            patch_img = img[y:y+patch_size, x:x+patch_size]
            if np.all(patch_mask > 0) and np.all(patch_img > 0):
                return (x, y)
    return None

patch_brain_pos = find_patch_in_brain(img_u8, sig_mask.astype(np.uint8), patch_size=10)
img_color_brain = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)

if patch_brain_pos:
    x, y = patch_brain_pos
    cv2.rectangle(img_color_brain, (x, y), (x+10, y+10), (255, 0, 0), 2)
    print(f'Parche de señal en cerebro: x={x}, y={y}')

    # --- Calcular SNR local del parche ---
    patch_vals = img_u8[y:y+10, x:x+10].ravel()
    snr_patch = np.mean(patch_vals) / sigma_est
    print(f"SNR de parche cerebral (simple) = {snr_patch:.3f}")
else:
    print("No se encontró un parche válido dentro del cerebro.")


print(f"Media señal (mu) = {mu_signal:.4f}")
print(f"SNR simple (mu / sigma) = {snr_simple:.3f}")
print(f"SNR corregido (Rician) = {snr_rician_corrected:.3f}")

# --- Visualización de parche ---
position = find_patch_near_brain_border(img_u8, sig_mask.astype(np.uint8))
img_color = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
if position:
    x, y = position
    cv2.rectangle(img_color, (x, y), (x+10, y+10), (0, 255, 0), 2)
    print(f'Valor de x: {x}, y: {y}')
else:
    print("No se encontró un parche válido cercano al borde cerebral.")

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.imshow(sig_mask, cmap='gray')
plt.title('Máscara Cerebral (Suave y Permisiva)')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
plt.title('Zona 10x10 en Background Cercano')
plt.axis('off')
plt.tight_layout()
plt.show()



# --- Mostrar comparación ---
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.imshow(sig_mask, cmap='gray')
plt.title('Máscara Cerebral (Suave y Permisiva)')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(img_color_brain, cv2.COLOR_BGR2RGB))
plt.title('Zona 10x10 en Señal Cerebral')
plt.axis('off')
plt.tight_layout()
plt.show()


# --------------------------
# BLOQUE: Aplicar ruido Gaussiano
# --------------------------
import numpy as np
from scipy.ndimage import rotate
import matplotlib.pyplot as plt

# Función para mostrar imágenes
def show(image, title):
    plt.figure(figsize=(6,6))
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# --- 1) Seleccionar rebanada ---
i = 14  # número de rebanada
img = rotate(data[i], 90, reshape=False)  # rotar como ejemplo
show(img, f'Original Rotated: {i}')

# --- 2) Aplicar ruido gaussiano ---
sigma = 0.05 * np.max(img)  # nivel de ruido relativo
noisy_img = img + sigma * np.random.randn(*img.shape)

# --- 3) Mostrar ---
show(noisy_img, f'Imagen con ruido Gaussiano σ={sigma:.2f}')
