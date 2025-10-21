import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from google.colab import drive
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
from scipy.signal import fftconvolve
drive.mount('/content/drive')
nii_path='/content/drive/MyDrive/LISA NII/LISA_0006_LF_axi.nii'


def show_image(img, title=""):
    plt.figure(figsize=(6,6))
    plt.imshow(img, cmap='gray', origin='lower')
    plt.title(title)
    plt.axis('off')
    plt.show()

def show_images_grid(images, titles=None, nrows=1, ncols=1, figsize=(12,6)):
    plt.figure(figsize=figsize)

    for i, img in enumerate(images):
        plt.subplot(nrows, ncols, i+1)
        plt.imshow(img, cmap='gray', origin='lower')
        if titles is not None:
            plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def norm01(x, eps=1e-12):
    x=vol[:,:,x]
    x = x.astype(np.float64)
    return (x - x.min()) / (x.max() - x.min() + eps)

def to_kspace(slice_img):
    K = np.fft.fft2(slice_img)
    K_shifted = np.fft.fftshift(K)
    K_magnitude = np.log1p(np.abs(K_shifted))
    K_phase = np.angle(K_shifted)
    return K, K_shifted, K_magnitude, K_phase

def gaussian_lpf_from_shape(kshape, sigma_pix):
    rows, cols = kshape
    y = np.arange(rows) - rows//2
    x = np.arange(cols) - cols//2
    X, Y = np.meshgrid(x, y)      # X -> cols, Y -> rows  => (rows, cols)
    R2 = X*X + Y*Y
    H = np.exp(-R2 / (2.0 * sigma_pix**2))
    return H


def gaussian_kernel_2d(size, sigma_pix):
    ax = np.arange(size) - size//2
    xx, yy = np.meshgrid(ax, ax)
    ker = np.exp(-(xx**2 + yy**2) / (2.0 * sigma_pix**2))
    ker /= ker.sum()
    return ker

# ***DESPLAZAMIENTO LINEAL DE N PÍXELES***

#***Desplazamiento lineal (Translational motion)***

#**Concepto:**
#Un desplazamiento lineal ocurre cuando el objeto se mueve constantemente durante la adquisición. Esto genera un desfase lineal en k-space, produciendo ghosting en la imagen reconstruida.

#**Ecuación:**


img = nib.load(nii_path)
vol = img.get_fdata()

slice_ids = 16
slice_img = norm01(slice_ids, eps=1e-12)

K, K_shifted, K_magnitude, K_phase = to_kspace(slice_img)

K_motion = K_shifted.copy()
filas, columnas = K_motion.shape
despl_pix = 50  # "50 píxeles de desplazamiento"
phase_row = np.exp(-2j * np.pi * np.arange(filas) * despl_pix / filas)[:, None]
K_motion *= phase_row  # multiplicación fila a fila

K_motion_magnitude = np.log1p(np.abs(K_motion))
K_motion_phase = np.angle(K_motion)

img_motion = np.fft.ifft2(np.fft.ifftshift(K_motion))
img_motion = np.abs(img_motion)

Images = [slice_img,img_motion,K_magnitude,K_motion_magnitude,K_phase,K_motion_phase]
Titulos = ["Slice Original","Imagen Reconstruida","En espacio K (Magnitud)","En espacio K con motion (Magnitud)","En espacio K (Fase)","En espacio K con motion (Fase)"]

show_images_grid(Images, Titulos, nrows=3, ncols=2, figsize=(6,6))


# ***DESPLAZAMIENTO ALEATORIO***

#***Desplazamiento aleatorio (Random motion)***

#**Concepto:**
#Simula temblores, respiración irregular u otros movimientos impredecibles. Cada línea de k-space recibe un desfase distinto, produciendo ghosting difuso.

#*Ecuación:**

#![image.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAn0AAABLCAYAAAAFzzFUAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAACQgSURBVHhe7Z0FuBzVGYZPEtzdLbiHIMUdirs7xSlSpLg7lFIIlGJFGjxosOCB4EkpVtzd3a0p75/5b04mM7uzu7PJ3bvf+zzz3LuzszNnzpyd8833/+dst549ew4NQgghhBCiS9M9+SuEEEIIIbowEn1CCCGEEG2ARJ8QQgghRBsg0SeEEEII0QZI9AkhhBBCtAESfUIIIYQozF577RVWW2215FVt7LjjjmGHHXZIXuWz+eabh7XWWit5JcpCok8IIYRoAtNMM01YeeWVw/jjj9/xetdddw1HHHFEWHDBBW1dUQ4//PBw2223hf79+4flllsuWVsbF1xwQbjvvvtC7969kzW1c+qpp4aFF144PPjgg/a61nINGjQorLLKKmGrrbZK1owM9fX73/8+9OjRI1kz/Dgsp5xySrI2hO222y7cdNNNtv7mm282USnykegTQgghmsABBxwQTjvtNBN+K6ywQrjoootMMM0999zhqquuMgFVlOOPPz58/vnnYbzxxguPP/54srY466yzjh17+umnD+uvv36ytjY22WQTE6v/+te/wrfffmvrai3Xyy+/bMJv6623zhWfa6yxRhhrrLHCPffck6wZdpzBgweHmWeeOay44opWn3DttdeGgQMHmkBE1F544YW2XmQj0SeEEEI0gXPPPTfsvPPO5kRtsMEGJoxuv/12c7mee+45E4NrrrlmsnVlevbsGaabbrrw+uuvdwiuWuA4iC0EGoKpHrdv9dVXD++88465hU495br00kvD0KFDc8XnkksuaeIwvb9JJ500PPbYY2GiiSayugO2mWSSSaw+qWdRGYk+IURppMNZrcgcc8zR4SJ0RehQaw0tdiZapY3RjnDVhgwZYq9//fXXMMUUU4TJJ5/cXr/yyismAmeffXZ7XY0FFljAxM4bb7xhr2u5jrh80047bejbt6+JJuowT2ziorFv/9+Psdhii4VZZ501/Pe//7XXTj3lQqhx/lnCExE51VRThfvvvz9ZMwzWTznllBbG/fjjj+04rAO2f+GFF+x/URmJPiFEKdDJ/f3vfzc3IOuJn5s0nQ8dTrOh01l33XWtTPWw77771hR6axX+9Kc/haOPPjpMNtlkyZrhNFpntUAboC24uKiFr7/+Omy77bbh7LPP7rTCDyfvxBNPDMcee2zo06ePrdtvv/3C/PPPH/75z3/aa8TeV199FZ5++mm7Lk899VR47bXXRlheffVVCwkDn/3f//5nouu8884Lhx12WDjjjDNMjFUDkfzMM8+EJ554Itxyyy12XNbF15r/L7nkkrDZZpuF4447zsTVTjvtFM455xw7n1lmmSWMM844HeLOqbdcb775ponF9Hbk8v38888juIkwzzzzmHAmnPuf//zH3EXuNQi/iSee2ESkqI5EnxBNhBsmT6DxTZybIzdGILfn7rvvHuF9kpFbERKtxxhjDBMVMUceeaSFXi6//PJwyCGHhBlnnDF5p3x2220369huuOEG67joKGqFsNI//vGPsPjii4c///nPydrRh9eftxEWXiOwnWuuucbajr9Ph5nuTMnHokOnbuIOtYw6K8oiiywS7rrrrvDAAw+YMFh77bWTd4rDAwX5XXT6xxxzTLK2c8GABr7X0K1bN/sbs/3225tYIRyJe7XUUkvZiNhbb73V6uWOO+6wv7PNNlvHSNd55503fPfdd5bv9tBDD4VffvnFlk8++cTez4NRtgg2RByw70cffdRcSELODqNlaTu0hw8++MBy5AjZjjvuuOGnn34KM8wwQ/jxxx/tvZh6y4VQHHPMMU34xeAcIlDTLLTQQuHDDz+0/8n1oyw48rQpjhfn/4l8JPqEaCK777572GabbexG+cMPP4S//OUv9mR8wgkn2PsIQm5Wn376qYkinlxxQFoNOgpu/jyFp10+3I6jjjrKOob33nuvI9zVDMihovMk/PPZZ59ldh5FoGMkvESnWI8bVSbUH3X74osvWkdJPhSv99xzz2SLEAYMGGBtCNFAnhSdYVzPOGIkzr/99ttWRzFl1VkRSPRfddVVLSH/+++/t3OqB4Q5InXZZZc1sTKqoJ4QrA8//HDmQpvBHcPNQ9BQ75xrDNcG5wzBh3hFsCPGcTARstQJ4UqulYNARKTxvodSuc64YgizSrDNSy+9FB555JFkTQh33nmnfU95z0OkDCw5//zzO45FOXiQY/AHZcyikXKxLYI4Fn2Ee3Ht0i4fTD311OHdd9+1/6m7Z599Nsw555xh+eWXt7YviiHRJ0STwTnhxvbFF19Y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)


img=nib.load(nii_path)
vol=img.get_fdata()

slice_ids=16
slice_img = norm01(slice_ids, eps=1e-12)
K, K_shifted, K_magnitude, K_phase = to_kspace(slice_img)


K_motion = np.copy(K_shifted)
filas, columnas = K_motion.shape
for i in range(filas):
    random_shift = np.random.uniform(-0.3, 0.3)  # movimiento aleatorio por línea
    phase_shift = np.exp(-2j * np.pi * i * random_shift / filas)
    K_motion[i, :] *= phase_shift
K_motion_magnitude = np.log1p(np.abs(K_motion))
#show_image(K_motion_magnitude, title="Espacio K con Motion (Magnitud)")
K_motion_phase = np.angle(K_motion)
#show_image(K_motion_phase, title="Espacio K con Motion (Fase)")

K_motion_ishift = np.fft.ifftshift(K_motion)
img_motion = np.fft.ifft2(K_motion_ishift)
img_motion = np.abs(img_motion)
#show_image(img_motion,title="Imagen Reconstruida")

Images = [slice_img, img_motion, K_magnitude, K_motion_magnitude, K_phase, K_motion_phase]
Titulos = ["Slice Original", "Imagen Reconstruida","En espacio K (Magnitud)","En espacio K con motion (Magnitud)","En espacio K (Fase)", "En espacio K con motion (Fase)"]

show_images_grid(Images, Titulos, nrows=3, ncols=2, figsize=(6,6))

# ***DESPLAZAMIENTO CON OSCILACIÓN***

#**Desplazamiento periódico (Oscillatory / Respiratory motion)**

#**Concepto:**
#Simula movimientos cíclicos, como la respiración. El objeto se desplaza siguiendo un patrón sinusoidal durante la adquisición, causando ghosts periódicos en la reconstrucción.

#**Ecuación:**

#![image.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmUAAACTCAYAAADP224KAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAADDiSURBVHhe7Z0LuFRV+YcXIogoKAKChAYoonKLkLhoKJmBESEhUEKpeIkiSkKLJA1JlCwkQ5JAkQxMLoZIEGjeULmIhIBmeEEyJIjwAoqCkv//u9jrsM5mz8yeOTPn7OH83ufZz8xe+7Yue/b6re/71p4qTZs2/cQIIYQQQogK5bDgUwghhBBCVCASZUIIIYQQCUCiTAghhBAiAUiUCSGEEEIkAIkyIYQQQogEIFEmhBBCCJEAJMqEEEIIIRKARJkQQgghRAKQKBNCCCGESAASZUIIIYQQCUCiTAghhBAiAUiUCSGEEEIkAIkyIYQQohIybNgw071792CtYrn88stN586dg7XMnHbaaeb66683zZs3D1Kiadq0qfn+978frCUfiTIhhBCVEjr0c889N1gzpk2bNraj//a3v20aNmwYpGaGjn/atGlm8eLF5ne/+12QmhuIpGeeecYMHz48SEnPUUcdZb773e+aPn36BCnx+OUvf2k++9nPmqefftquU17KTfmph2z46U9/ahYtWmTmz59vunbtGqTGp127drYdPvzwQ7vu6pNz+stDDz1kbrzxRpvXf/zjH+aDDz4wP/rRj2wdpKJHjx7m5JNPDtaMFX6zZs2y53vggQdKRCnnoO3cte66666s7oF8IVEmhBCiUkIHf/PNN5sOHTqYH/7wh1ao1K1b1/Tv398sXLjQDBw4MNgzPa+//roVEccee6zZuXNnkJobF198sc1DXL70pS+Z73znO+aKK64IUjLTr18/K7x+//vfm/fff98KIvKPSMMCdf/999u6iMtNN91k3n77bVOzZk2zevXqIDU+XP/dd981a9assevU5+DBg82uXbtMnTp1zK233mrzTH4///nPm7vvvtsKtzvvvNMcd9xxVpilom3btlbAOZYvX25+/vOfm+rVq9s6+PKXv2zTqYdJkyZZoff3v//d/OIXvzBbt26128oTiTIhhBCVknHjxpkhQ4aY//73v9Ziwuftt99uhdr//vc/M2DAgGDPzDRr1sxUq1bNvPrqq0FK9lx66aWmUaNG5pNPPjENGjQIUtPz8MMP23IgIuKC9Wjz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)


img=nib.load(nii_path)
vol=img.get_fdata()

slice_ids=16
slice_img = norm01(slice_ids, eps=1e-12)
K, K_shifted, K_magnitude, K_phase = to_kspace(slice_img)


K_motion = np.copy(K_shifted)
filas, columnas = K_motion.shape
for i in range(filas):
    periodic_shift = 3 * np.sin(2 * np.pi * i / 4)  # amplitud 3, periodo 4 líneas
    phase_shift = np.exp(-2j * np.pi * i * periodic_shift / filas)
    K_motion[i, :] *= phase_shift
K_motion_magnitude = np.log1p(np.abs(K_motion))
#show_image(K_motion_magnitude, title="Espacio K con Motion (Magnitud)")
K_motion_phase = np.angle(K_motion)
#show_image(K_motion_phase, title="Espacio K con Motion (Fase)")

K_motion_ishift = np.fft.ifftshift(K_motion)
img_motion = np.fft.ifft2(K_motion_ishift)
img_motion = np.abs(img_motion)
#show_image(img_motion,title="Imagen Reconstruida")

Images = [slice_img, img_motion, K_magnitude, K_motion_magnitude, K_phase, K_motion_phase]
Titulos = ["Slice Original", "Imagen Reconstruida","En espacio K (Magnitud)","En espacio K con motion (Magnitud)","En espacio K (Fase)", "En espacio K con motion (Fase)"]

show_images_grid(Images, Titulos, nrows=3, ncols=2, figsize=(6,6))

# ***BLUR + Filtrado Low-Pass***
#Un filtro paso bajo (Low-Pass) atenúa las frecuencias espaciales altas, que son responsables de los detalles finos y bordes de la imagen.
#En RM, aplicar un filtro de este tipo sobre la magnitud del espacio K simula una pérdida de resolución espacial o difuminado (blur), sin alterar la información estructural codificada en la fase.

#El filtro paso bajo gaussiano se define como:
#**Referencias bibliográficas**

#*   Gonzalez, R. C., & Woods, R. E. (2018). Digital Image Processing (4th ed.). Pearson.
#*   Liang, Z.-P., & Lauterbur, P. C. (2000). Principles of Magnetic Resonance Imaging: A Signal Processing Perspective. IEEE Press.



img=nib.load(nii_path)
vol=img.get_fdata()

slice_ids=16
slice_img = norm01(slice_ids, eps=1e-12)
K, K_shifted, K_magnitude, K_phase = to_kspace(slice_img)

sigma_k = 30   #Con sigmas muy bajo se difumina demasiado
H = gaussian_lpf_from_shape(K_shifted.shape, sigma_k)

K_filtered = K_shifted * H
K_filtered_mag_log = np.log1p(np.abs(K_filtered))

img_blur = np.fft.ifft2(np.fft.ifftshift(K_filtered))
img_blur = np.abs(img_blur)  # o np.real(img_blur)

Imagenes = [slice_img,img_blur,K_magnitude, K_filtered_mag_log,K_phase,np.angle(K_filtered)]

Titulos = ["Slice Original","Imagen Reconstruida","Magnitud FFT Original","Magnitud FFT Filtrada","Fase FFT Original","Fase FFT Filtrada"]

show_images_grid(Imagenes, Titulos, nrows=3, ncols=2, figsize=(10,10))

# ***BLUR + RUIDO POR BOBINA (SENSE)***

img=nib.load(nii_path)
vol=img.get_fdata()

slice_ids=16
slice_img = norm01(slice_ids, eps=1e-12)
K, K_shifted, K_magnitude, K_phase = to_kspace(slice_img)

ny, nx = slice_img.shape
x, y = np.meshgrid(np.linspace(-1,1,nx), np.linspace(-1,1,ny))

S1 = np.exp(-((x-0.5)**2 + (y+0.5)**2))
S2 = np.exp(-((x+0.5)**2 + (y+0.5)**2))
S3 = np.exp(-((x+0.5)**2 + (y-0.5)**2))
S4 = np.exp(-((x-0.5)**2 + (y-0.5)**2))
S_coils = [S1, S2, S3, S4]

coil_imgs = []
K_coils = []
noise_std = 7

for S in S_coils:
    I_coil = slice_img * S
    K_coil = np.fft.fftshift(np.fft.fft2(I_coil))
    noise = noise_std * (np.random.randn(*K_coil.shape) + 1j*np.random.randn(*K_coil.shape))
    K_noisy = K_coil + noise
    I_noisy = np.abs(np.fft.ifft2(np.fft.ifftshift(K_noisy)))
    coil_imgs.append(I_noisy)
    K_coils.append(K_noisy)

# --- Reconstrucción SENSE (combinación ponderada) ---
numerator = np.zeros_like(slice_img, dtype=complex)
denominator = np.zeros_like(slice_img)

for S, I in zip(S_coils, coil_imgs):
    numerator += np.conj(S) * I
    denominator += np.abs(S)**2

recon_sense = np.abs(numerator / (denominator + 1e-8))

# --- FFT de la reconstrucción ---
K_recon = np.fft.fftshift(np.fft.fft2(recon_sense))
K_recon_mag = np.log1p(np.abs(K_recon))
K_recon_phase = np.angle(K_recon)

# --- Mostrar resultados ---
imagenes = [slice_img, recon_sense,np.log1p(K_magnitude),K_recon_mag, K_phase, K_recon_phase, S1,S2,S3, S4]

titulos = ["Imagen Original","Reconstrucción SENSE (ruido bobinas)","Magnitud FFT Original", "Magnitud FFT Reconstruida","Fase FFT Original","Fase FFT Reconstruida","Bobina 1", "Bobina 2","Bobina 3", "Bobina 4"]

show_images_grid(imagenes, titulos, nrows=5, ncols=2, figsize=(12,12))

# ***BLUR POR MOVIMIENTO***
#El blur por movimiento ocurre porque:

#1.   Durante la adquisición, cada línea de phase-encode se registra en un momento distinto.
#2.   Si el objeto se mueve, la fase de k-space cambia según la línea.
#3.   Esto genera incoherencia de fase que, al reconstruir con IFFT, se traduce en desenfoque difuso (blur) en la imagen.

img=nib.load(nii_path)
vol=img.get_fdata()

slice_ids=16
slice_img = norm01(slice_ids, eps=1e-12)

K = np.fft.fft2(slice_img)
K_shifted = np.fft.fftshift(K)
K_mag = np.abs(K_shifted)
K_phase = np.angle(K_shifted)

filas, columnas = K_shifted.shape
v = np.linspace(-1, 1, filas)  # eje phase-encode
phi_motion = 4 * np.sin(2 * np.pi * v)  # desfase suave, ajustable
phi_matrix = np.tile(phi_motion[:, np.newaxis], (1, columnas))  # aplicar a cada columna

K_motion = K_mag * np.exp(1j * (K_phase + phi_matrix))

img_blur = np.fft.ifft2(np.fft.ifftshift(K_motion))
img_blur = np.abs(img_blur)

images = [slice_img,img_blur,np.log1p(K_mag),np.log1p(np.abs(K_motion)),K_phase,np.angle(K_motion)]

titles = ["Imagen Original","Imagen con Blur (fase modificada)","Magnitud K Original","Magnitud K con Blur","Fase K Original","Fase K con Blur"]

show_images_grid(images, titles, nrows=3, ncols=2, figsize=(12,10))

nii_path1='/content/drive/MyDrive/LISA NII/LISA_0035_LF_axi.nii'
img=nib.load(nii_path1)
vol=img.get_fdata()
zooms=img.header.get_zooms()
slice_ids=14
slice_img=vol[:,:,slice_ids]
slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-12) #normalizado
dx1, dy1 = zooms[0], zooms[1]
#show_image(slice_img, dx1, dy1, title=f"Slice {slice_ids}")
show_image(slice_img, title=f"Slice {slice_ids}")

nii_path2='/content/drive/MyDrive/LISA NII/LISA_0017_LF_axi.nii'
img=nib.load(nii_path2)
vol=img.get_fdata()
zooms=img.header.get_zooms()
slice_ids=16
slice_img=vol[:,:,slice_ids]
slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-12)
dx2, dy2 = zooms[0], zooms[1]
#show_image(slice_img, dx2, dy2, title=f"Slice {slice_ids}")
show_image(slice_img, title=f"Slice {slice_ids}")

