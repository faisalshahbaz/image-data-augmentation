        
import cv2
import random
import numpy as np 
import os

class Data_augmentation:
    def __init__(self, path, image_name):
       
        self.path = path
        self.name = image_name
        print(path+image_name)
        self.image = cv2.imread(path+image_name)

    def rotate(self, image, angle=90, scale=1.0):
       
        w = image.shape[1]
        h = image.shape[0]
        #rotate matrix
        M = cv2.getRotationMatrix2D((w/2,h/2), angle, scale)
        #rotate
        image = cv2.warpAffine(image,M,(w,h))
        return image

    def flip(self, image, vflip=False, hflip=False):
      
        if hflip or vflip:
            if hflip and vflip:
                c = -1
            else:
                c = 0 if vflip else 1
            image = cv2.flip(image, flipCode=c)
        return image 
    def brightness_augment(self,img, factor=0.5): 
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) #convert to hsv
        hsv = np.array(hsv, dtype=np.float64)
        hsv[:, :, 2] = hsv[:, :, 2] * (factor + np.random.uniform()) #scale channel V uniformly
        hsv[:, :, 2][hsv[:, :, 2] > 255] = 255 #reset out of range values
        img = cv2.cvtColor(np.array(hsv, dtype=np.uint8), cv2.COLOR_HSV2RGB)
        return img
    def noisy(self, img, noise_type="gauss"):
        
        if noise_type == "gauss":
            image=img.copy() 
            mean=0
            st=0.7
            gauss = np.random.normal(mean,st,image.shape)
            gauss = gauss.astype('uint8')
            image = cv2.add(image,gauss)
            return image

        elif noise_type == "sp":
            image=img.copy() 
            prob = 0.05
            if len(image.shape) == 2:
                black = 0
                white = 255            
            else:
                colorspace = image.shape[2]
                if colorspace == 3:  # RGB
                    black = np.array([0, 0, 0], dtype='uint8')
                    white = np.array([255, 255, 255], dtype='uint8')
                else:  # RGBA
                    black = np.array([0, 0, 0, 255], dtype='uint8')
                    white = np.array([255, 255, 255, 255], dtype='uint8')
            probs = np.random.random(image.shape[:2])
            image[probs < (prob / 2)] = black
            image[probs > 1 - (prob / 2)] = white
            return image
    def add_GaussianNoise(self, img, fsize = 9):
            #image=img.copy()
            return cv2.GaussianBlur(img, (fsize, fsize), 0)
    def image_augment(self, save_path): 

        img = self.image.copy()
        img_flip = self.flip(img, vflip=True, hflip=False)
        img_rot = self.rotate(img)
        img_nos = self.noisy(img, noise_type="gauss")
        img_gaussian = self.add_GaussianNoise(img)
        
        name_int = self.name[:len(self.name)-4]
        cv2.imwrite(save_path+'%s' %str(name_int)+'_vflip.jpg', img_flip)
        cv2.imwrite(save_path+'%s' %str(name_int)+'_rot.jpg', img_rot)
        cv2.imwrite(save_path+'%s' %str(name_int)+'_GaussianNoise.jpg', img_gaussian)
        cv2.imwrite(save_path+'%s' %str(name_int)+'_nos.jpg', img_nos)
        cv2.imwrite(save_path+'%s' %str(name_int)+'_GaussianNoise.jpg', img_gaussian)
    
def main(file_dir,output_path):
    for root, _, files in os.walk(file_dir):
        print(root)
    for file in files:
        raw_image = Data_augmentation(root,file)
        raw_image.image_augment(output_path)
    