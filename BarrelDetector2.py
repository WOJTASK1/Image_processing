import cv2
import numpy as np
from cv2.typing import MatLike
from skimage import measure
import skimage.io as sk
from DetectedObjects import DetectedObjects
import os
from test_visoncore import BarrelDetectionConfig,Config

class BarrelDetector:

    def treshold_hsv(self) -> None:
        if self.img is None or not isinstance(self.img, np.ndarray):
            raise ValueError("detect_barrels: mask and img cannot be None")
        
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array(BarrelDetectionConfig.LOWER_HSV)
        upper_hsv = np.array(BarrelDetectionConfig.UPPER_HSV)
        self.mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
       
    
    def clean_mask(self, kernel_size: int = 7) -> None:
        if self.mask is None or not isinstance(self.mask, np.ndarray):
            raise ValueError("clean_mask: mask must be a valid np.ndarray")
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        cleaned = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, kernel)
        self.mask = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)    
 

    def detect_barrels(self,image : MatLike)->list:
        self.img = image
        self.mask = None
        self.treshold_hsv()
        self.clean_mask(BarrelDetectionConfig.KERNEL_SIZE)

        labeled_mask = measure.label(self.mask, connectivity=2)
        regions = measure.regionprops(labeled_mask)
        
        if len(regions) == 0:
            print("None of the barrels have been detected!")
            return []

        barrels = []

        for region in regions:
            cy, cx = region.centroid
            centroid = (int(cx), int(cy))
            #print(f"Detected barrel at centroid: {centroid}, area: {region.area}")
            region_mask = (labeled_mask == region.label).astype(np.uint8) * 255
            if region.area < BarrelDetectionConfig.MINIMAL_AREA or region.area > BarrelDetectionConfig.MAXIMAL_AREA:
                continue
                
            contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                contour = contours[0]
                barrels.append(DetectedObjects(centroid, contour))
      
        return barrels
    
    def draw_barrels(self,barrels) -> np.ndarray:
        
        output_img = self.img.copy()
        if barrels is not None:
            for barrel in barrels:

                if barrel.contour is None or barrel.centroid is None:
                    continue
                cv2.drawContours(output_img, [barrel.contour], -1, (255, 0, 255), 10)
                cv2.circle(output_img, barrel.centroid, 25, (255, 255, 255), -1)
        output_img = cv2.resize(output_img, (0, 0), fx=1/5, fy=1/5)
        cv2.imshow("Detected Barrels", output_img)
        cv2.waitKey(0)




def readImagesFromFolder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".JPG"):
            img_path = os.path.join(folder_path, filename)
            images.append(img_path)
    return images

if __name__ == '__main__':
    Config.load("PipeBarrelDetector")
    try:
        path = "rozne/fake.JPG"
        folder_path = "/home/wojtek/Documents/uczelnia/RAPTORS/Droniada/beczki"
      
        images = readImagesFromFolder(folder_path)
        images.append(path)
        detector = BarrelDetector() 
        
        for img_path in images:
            print("#########\nWykrywanie beczek w obrazie:", img_path)
            img = cv2.imread(img_path)
            if img is None:
                print("Nie można wczytać obrazu:", img_path)
                continue
            result = detector.detect_barrels(img)
    
            detector.draw_barrels(result)
            if result is None or len(result) == 0:
                print("Brak beczek w obrazie:", img_path)
                continue
      
            for i, barrel in enumerate(result, 1):
                print(f"Beczka {i}: centroid = {barrel.centroid}, kontur (wymiary) = {barrel.contour.shape}")
            
    except Exception as e:
        print("Wystąpił błąd:", e)
    finally:
        cv2.destroyAllWindows
        cv2.waitKey(1)

    