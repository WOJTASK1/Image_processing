import cv2
import numpy as np
from skimage import measure
import skimage.io as sk
import matplotlib.pyplot as plt
from DetectedObjects import DetectedObjects
import os

class BarrelDetector:
    
    mask = None
    img = None
    MINIMAL_BARREL_AREA = 9000
    MAXIMAL_BARREL_AREA = 35000
    
    def __init__(self, path : str) -> None:
        #self.img = sk.imread(path)
        return
        
    def setImage(self, path : str) -> None:
        if path is None or not isinstance(path, str):
            raise ValueError("setImage: path must be a valid string")
        self.img = sk.imread(path)
        self.mask = None  # Reset mask when a new image is set

    
    def treshold_hsv(self) -> None:
        if self.img is None or not isinstance(self.img, np.ndarray):
            raise ValueError("detect_barrels: mask and img cannot be None")
        
        hsv = cv2.cvtColor(self.img, cv2.COLOR_RGB2HSV)
        lower_hsv = np.array([48, 50, 30])
        upper_hsv = np.array([159, 255, 190])
        self.mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
       
    
    def clean_mask(self, kernel_size: int = 7) -> None:
        if self.mask is None or not isinstance(self.mask, np.ndarray):
            raise ValueError("clean_mask: mask must be a valid np.ndarray")
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        cleaned = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, kernel)
        self.mask = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)    
 

    def detect_barrels(self,kernel_size: int = 7)->list:
        
        self.treshold_hsv()
        self.clean_mask(kernel_size)

        labeled_mask = measure.label(self.mask, connectivity=2)
        regions = measure.regionprops(labeled_mask)
        #regions = [r for r in regions if self.MINIMAL_BARREL_AREA < r.area < self.MAXIMAL_BARREL_AREA]
        if len(regions) == 0:
            print("None of the barrels have been detected!")
            return []

        barrels = []

        for region in regions:
            cy, cx = region.centroid
            centroid = (int(cx), int(cy))
            #print(f"Detected barrel at centroid: {centroid}, area: {region.area}")
            region_mask = (labeled_mask == region.label).astype(np.uint8) * 255
            if region.area < self.MINIMAL_BARREL_AREA or region.area > self.MAXIMAL_BARREL_AREA:
                continue
                
            contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                contour = contours[0]
                barrels.append(DetectedObjects(centroid, contour))
      
        return barrels
    
    def draw_barrels(self,barrels,path) -> np.ndarray:
        
        output_img = self.img.copy()
        if barrels is not None:
            for barrel in barrels:

                if barrel.contour is None or barrel.centroid is None:
                    continue
                cv2.drawContours(output_img, [barrel.contour], -1, (255, 0, 0), 2)
                cv2.circle(output_img, barrel.centroid, 25, (255, 255, 255), -1)
            
        plt.figure(figsize=(10, 5))
        plt.imshow(output_img)
        plt.title(f'Wykryte beczki i kontury+{path}')
        plt.axis('off')
        plt.show()




def readImagesFromFolder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".JPG"):
            img_path = os.path.join(folder_path, filename)
            #img = sk.imread(img_path)
            images.append(img_path)
    return images

if __name__ == '__main__':
    try:
        path = "rozne/fake.JPG"
        folder_path = "/home/wojtek/Documents/uczelnia/RAPTORS/Droniada/beczki"
      
        images = readImagesFromFolder(folder_path)
        images.append(path)
        detector = BarrelDetector(path) 
        
        print("Liczba obrazów w folderze:", len(images))
        for img_path in images:
            print("#########\nWykrywanie beczek w obrazie:", img_path)
            detector.setImage(img_path)
            
            result = detector.detect_barrels(kernel_size=10)
    
            detector.draw_barrels(result, img_path)
            if result is None or len(result) == 0:
                print("Brak beczek w obrazie:", img_path)
                continue
      
            for i, barrel in enumerate(result, 1):
                print(f"Beczka {i}: centroid = {barrel.centroid}, kontur (wymiary) = {barrel.contour.shape}")
            
    except Exception as e:
        print("Wystąpił błąd:", e)


    