from visioncore import Config

class BarrelDetectionConfig:

    MINIMAL_AREA = 9000
    MAXIMAL_AREA = 35000
    
    #tresholds:
    LOWER_HSV = [48, 50, 30]
    UPPER_HSV = [159, 255, 190]
    # Morphological operations
    KERNEL_SIZE = 10

class PipeDetectionConfig:
    MINIMAL_AREA = 25000
    MAXIMAL_AREA = 4800000
    
    #tresholds:
    LOWER_HSV = [0, 0, 179]
    UPPER_HSV = [179, 226, 255]
    # Morphological operations
    KERNEL_SIZE = 10

Config.load("PipeBarrelDetector")
Config.register(BarrelDetectionConfig)
Config.register(PipeDetectionConfig)
#Config.save("PipeBarrelDetector")