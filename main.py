import numpy as np
import cv2
from scipy import fftpack

ADDRESS = "./samurai2.jpg"
BLOCK_SIZE = 8
THRESH = 0.0003

def ImportImage(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR).astype(float)
    return image


#Assumption is that the input colorspace is BGR or YCC only
def ConvertColorspace(image, conversionType):

    if conversionType == "YCC":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    elif conversionType == "BGR":
        image = cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)
    else:
        printf("Unable to convert")

    return image


def ShowImage(image):
    colorOne = image
    colorTwo = cv2.cvtColor(colorOne[:,:,0], cv2.COLOR_GRAY2RGB)
    colorThree = cv2.cvtColor(colorOne[:,:,1], cv2.COLOR_GRAY2RGB)
    colorFour = cv2.cvtColor(colorOne[:,:,2], cv2.COLOR_GRAY2RGB)

    topLayer = np.concatenate((colorOne, colorTwo), axis=0)
    bottomLayer = np.concatenate((colorThree, colorFour), axis=0)
    finalImage = np.concatenate((topLayer, bottomLayer), axis=1)
    
    print("Showing Image of Size: ", image.shape)
    cv2.namedWindow("Imshow Result", cv2.WINDOW_NORMAL)
    cv2.imshow("Imshow Result", finalImage)
    cv2.waitKey()


def CropImage(image):
    dim0 = (image.shape[0]//8)*8
    dim1 = (image.shape[1]//8)*8
   
    if (len(image.shape)==3):
        cropImage = image[:dim0,:dim1,:]
    else:
        cropImage = image[:dim0,:dim1]
   
    return cropImage

def BlockDCT(block):
   
    dct = np.zeros(block.shape)

    for i in range(3):
        dct[:,:,i] = fftpack.dct(
                     fftpack.dct(block[:,:,i], axis=0, norm="ortho"),
                     axis=1, norm="ortho")

    return dct

def BlockIDCT(block):


    idct = np.zeros(block.shape)
    
    for i in range(3):
        idct[:,:,i] = fftpack.idct(
                      fftpack.idct(block[:,:,i], axis=0, norm="ortho"),
                      axis=1, norm="ortho")
    
    return idct




def DCTTransform(image, n=BLOCK_SIZE, inverse=False):

    dim = image.shape
    dctImage = np.zeros(dim)

    #Convert the image in nxn blocks
    
    if inverse:
        print("Inversing the DCT")
    else:
        print("DCTing")

    for i in range(0, dim[0],n):
        for j in range(0, dim[1],n):
            if inverse: 
                dctImage[i:i+n,j:j+n,:] = BlockIDCT(image[i:i+n, j:j+n, :])
            else:
                dctImage[i:i+n,j:j+n,:] = BlockDCT(image[i:i+n, j:j+n, :])
    


    
    print("DCT TRansform size: ", dctImage.shape)
    return dctImage

def ThresholdImage(image):

    
    bMax = image[:,:,0].max()
    gMax = image[:,:,1].max()
    rMax = image[:,:,2].max()

    print("Thresholding params: ", THRESH*bMax, THRESH*gMax, THRESH*rMax)
  

    
    image[:,:,0] = image[:,:,0] * (abs(image[:,:,0]) > THRESH*bMax)
    image[:,:,1] = image[:,:,1] * (abs(image[:,:,1]) > THRESH*gMax)
    image[:,:,2] = image[:,:,2] * (abs(image[:,:,2]) > THRESH*rMax)
    
    
    



if __name__ == '__main__':
    image = ImportImage(ADDRESS)
    if (image.any()):
        print("Image Loaded. Shape: ", image.shape)
    else:
        print("Oopse! error")

    image = CropImage(image)
    
    #imageConverted = ConvertColorspace(image, "YCC")
    #ShowImage(imageConverted)
    dctImage = DCTTransform(image)

    ThresholdImage(dctImage)
    idctImage = DCTTransform(dctImage, inverse=True)
    #unCompImage = ConvertColorspace(unCompImage, "BGR")
    idctImage = np.asarray(idctImage, dtype=np.uint8)
    recoveredImage = ConvertColorspace(idctImage, "BGR")

    cv2.imshow("IDCT Image", idctImage)

    cv2.waitKey()


