from scipy import fftpack
import numpy as np
a = np.random.randint(0, 256, size=(8, 8, 3)) 

print(a[:,:,0])

z = fftpack.dct(fftpack.dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")

zc = zc = fftpack.idct(fftpack.idct(z, axis=0, norm="ortho"), axis=1, norm="ortho")

print(zc[:,:,0])

