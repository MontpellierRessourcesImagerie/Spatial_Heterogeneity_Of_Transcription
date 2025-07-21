from scipy.ndimage import median_filter


class MedianFilter:


    def __init__(self, image, radius=1, name=None):
        self.image = image
        self.radius = radius
        self.name = name
        self.result = None


    def run(self):
        self.result = median_filter(self.image, size=2*self.radius+1)


    def getResult(self):
        return self.result


    def getName(self):
        return self.name