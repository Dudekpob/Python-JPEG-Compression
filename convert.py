from scipy.fft import dct
import scipy.fftpack
from scipy.signal import convolve2d
import cv2
from PIL import Image
import numpy as np
import os
from multiprocessing.pool import Pool
import rawpy
import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt

QY= np.array([
        [16, 11, 10, 16, 24,  40,  51,  61],
        [12, 12, 14, 19, 26,  58,  60,  55],
        [14, 13, 16, 24, 40,  57,  69,  56],
        [14, 17, 22, 29, 51,  87,  80,  62],
        [18, 22, 37, 56, 68,  109, 103, 77],
        [24, 36, 55, 64, 81,  104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
        ])

QC= np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        ])

Q1= np.array([
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        ])


class ver1:
    pass

class ver2:
    def __init__(self):
        self.shape = None
        ###

def dct2(a):
    return scipy.fftpack.dct( scipy.fftpack.dct( a.astype(float), axis=0, norm='ortho' ), axis=1, norm='ortho' )

def idct2(a):
    return scipy.fftpack.idct( scipy.fftpack.idct( a.astype(float), axis=0 , norm='ortho'), axis=1 , norm='ortho')

def quantization(Q, type):
    if (type == 'lum'):
            return(np.divide(Q, QY).round().astype(int))
    elif (type == 'chr'):
            return(np.divide(Q, QC).round().astype(int))
    elif (type == 'nul'):
            return(np.divide(Q, Q1).round().astype(int))
    else:
            raise ValueError("Type choice %s unknown" %(type))


def dequantization(Q, type):
    if (type == 'lum'):
            return(np.multiply(Q, QY))
    elif (type == 'chr'):
            return(np.multiply(Q, QC))
    elif (type == 'nul'):
            return(np.multiply(Q, Q1))
    else:
            raise ValueError("Type choice %s unknown" %(type))


def zigzag(A):
    template= n= np.array([
            [0,  1,  5,  6,  14, 15, 27, 28],
            [2,  4,  7,  13, 16, 26, 29, 42],
            [3,  8,  12, 17, 25, 30, 41, 43],
            [9,  11, 18, 24, 31, 40, 44, 53],
            [10, 19, 23, 32, 39, 45, 52, 54],
            [20, 22, 33, 38, 46, 51, 55, 60],
            [21, 34, 37, 47, 50, 56, 59, 61],
            [35, 36, 48, 49, 57, 58, 62, 63],
            ])
    if len(A.shape)==1:
        B=np.zeros((8,8))
        for r in range(0,8):
            for c in range(0,8):
                B[r,c]=A[template[r,c]]
    else:
        B=np.zeros((64,))
        for r in range(0,8):
            for c in range(0,8):
                B[template[r,c]]=A[r,c]
    return B


class RLE:
    def _init_(self):
        pass

    def encode(self, img):
        copy_img = img.copy()
        copy_img = copy_img.flatten()
        copy_shp = copy_img.shape

        length_sp = copy_shp[0] * 2 + len(copy_shp) +1
        space = np.empty(length_sp).astype(img.dtype)
        space[0] = len(copy_shp)

        newImg = space
        newImgSize = len(copy_shp) + 1

        idx = 0
        for shp in copy_shp:
            newImg[idx + 1] = shp
            idx += 1
        
        curr = 0

        while curr < copy_shp[0]:
            frag = copy_img[curr]
            check = 1

            while curr + check < copy_shp[0] and copy_img[curr + check] == frag:
                check += 1
            
            newImg[newImgSize] = check
            newImg[newImgSize + 1] = frag
            newImgSize += 2
            curr += check
        
        return newImg[:newImgSize]
    
    def decode(self, img):
        size = np.empty(img[0])
        shp = img[0]

        cnt = 1
        for i in range(0, img[0]):
            size[i] = img[i+1]
            cnt *= size[i]

        cnt = int(cnt)
        space = np.empty(cnt).astype(int.dtype)
        newImg = space

        curr = img[0] + 1
        newImgSize = 0

        while curr < shp:
            check = img[newImgSize]
            for i in range(0, check):
                newImg[newImgSize] = img[curr + 1]
                newImgSize += 1
            curr += 2
        
        return np.reshape(newImg, tuple(size.astype(int))).astype(img.dtype)



def compressLayer(Layer , Q):  

    img_weight, img_height = Layer.shape
    OutputLayer = []
    for i in range(0, img_weight, 8):
        for j in range(0, img_height, 8):
            result_d = (dct2(Layer[i:i+8,j:j+8]))
            result_q = (quantization(result_d, Q))
            result_z = (zigzag(result_q))
            OutputLayer.append(result_z)
    return   np.array(OutputLayer).reshape(-1)

def decompressLayer(CompressedLayer,  Q):
    
    OutputLayer = np.zeros((128, 128))
    for idx,i in enumerate(range(0,CompressedLayer.shape[0], 64)):
        x,y = idx//16, idx%16
        print(x,y)
        current_block = CompressedLayer[i:i+64]
        result_z = zigzag(current_block)
        result_q = (dequantization(result_z, Q))
        result_id = (idct2(result_q))
        OutputLayer[x*8:(x*8+8), y*8:(y*8+8)] = result_id
    return OutputLayer


def chromaSubsampling(Q, type):               
    if type == '4:2:2':
        k = np.array([[0.5], [0.5]])
        Q = np.repeat(convolve2d(Q, k, mode='valid')[::2,:], 2, axis=0)    
    else:
        Q = Q
        return Q

    return Q  

data1=ver1()
data1.shape=(1,1)

data2=ver2()
data2.shape=(1,1)


RGB = cv2.imread("4.png")
RGB = RGB[170:298, 170:298]
plt.imshow(RGB)
plt.show()

    
# 1 Convert RGB to YCrCb
YCrCb=cv2.cvtColor(RGB,cv2.COLOR_RGB2YCrCb).astype(int)
plt.imshow(YCrCb)
plt.show()
# 2 Chroma Subsampling 

Ratio_1 = '4:2:2' # Subsampling 
Ratio_2 = '4:4:4' # 



Y = YCrCb[:,:,0]  - 128
Cr = YCrCb[:,:,1] - 128
Cb = YCrCb[:,:,2] - 128

Y = chromaSubsampling(Y, Ratio_1)
Cr = chromaSubsampling(Cr, Ratio_1)
Cb = chromaSubsampling(Cb, Ratio_1)
# 3 4 5 6 


Y_z = compressLayer(Y, 'nul')
Cr_z = compressLayer(Cr, 'nul')
Cb_z = compressLayer(Cb, 'nul')

# 7RLE

#rle_y = RLE()

#Y_encode = rle_y.encode(Y_z)
#Cr_encode = RLE.encode(Cr_z)
#Cb_encode = RLE.encode(Cb_z)

# 8 RLE

#Y_decode = rle_y.decode(Y_encode)
#Cr_decode = RLE.decode(Cr_encode)
#Cb_decode = RLE.decode(Cb_encode)

    # 9  Rev_ZigZag  10 dequantize  11 IDCT 


Y_rev = decompressLayer(Y_z, 'nul')
Cr_rev = decompressLayer(Cr_z, 'nul')
Cb_rev = decompressLayer(Cb_z, 'nul')


# 12 Chroma resampling


# 13 Conversion YCrCb na RGB
Y_rev = Y_rev + 128
Cr_rev = Cr_rev + 128
Cb_rev = Cb_rev + 128



YCrCb=np.dstack([Y_rev,Cr_rev,Cb_rev]).clip(0,255).astype(np.uint8)

# cv2.imshow("Image", YCrCb)
plt.imshow(YCrCb)
plt.show()
# obraz 128x128
RGB_decoded=cv2.cvtColor(YCrCb.astype(np.uint8),cv2.COLOR_YCrCb2RGB)

# plt.imshow(RBG)



imgplot2 = plt.imshow(RGB_decoded)

plt.show()

print("Done")

