import cv2
import numpy as np
from skimage import measure
import skimage.io as sk
import matplotlib.pyplot as plt
from DetectedObjects import DetectedObjects

class BarrelDetector:
    
    mask = None
    MINIMAL_BARREL_AREA = 15000
    MAXIMAL_BARREL_AREA = 40000
    
    def __init__(self, path : str) -> None:
        self.img = sk.imread(path)
        

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
        regions = [r for r in regions if self.MINIMAL_BARREL_AREA < r.area < self.MAXIMAL_BARREL_AREA]

        if len(regions) == 0:
            print("None of the barrels have been detected!")
            return self.img.copy(), []

        sorted_regions = sorted(regions, key=lambda r: r.area, reverse=True)
        barrels = []

        for region in sorted_regions:
            cy, cx = region.centroid
            centroid = (int(cx), int(cy))

            region_mask = (labeled_mask == region.label).astype(np.uint8) * 255
            contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                contour = contours[0]
                barrels.append(DetectedObjects(centroid, contour))
      
        return barrels
    
    def draw_barrels(self,barrels) -> np.ndarray:
        
        output_img = self.img.copy()
        
        for barrel in barrels:
            cv2.drawContours(output_img, [barrel.contour], -1, (255, 0, 0), 2)
            cv2.circle(output_img, barrel.centroid, 25, (255, 255, 255), -1)
            
        plt.figure(figsize=(10, 5))
        plt.imshow(output_img)
        plt.title("Wykryte beczki i kontury")
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    
    try:
        path = "rozne/fake.JPG"
        detector = BarrelDetector(path)
        
        result = detector.detect_barrels(100)
        
        detector.draw_barrels(result)
        
        for i, barrel in enumerate(result, 1):
            print(f"Beczka {i}: centroid = {barrel.centroid}, kontur = {barrel.contour.shape}")

    except Exception as e:
        print("Wystąpił błąd:", e)
