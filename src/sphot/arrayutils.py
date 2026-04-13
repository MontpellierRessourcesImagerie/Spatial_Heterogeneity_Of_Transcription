import numpy as np



class NDArrayUtil:


    @classmethod
    def resizeTo(cls, anArray, zz, yy, xx):
        """
        Resize the array to the given dimensions, cropping when it is larger and padding when it is smaller.
        :param anArray: numpy array
        :param xx: desired height
        :param yy: desired width
        :param zz: desired depth
        :return: resized array
        """
        d = anArray.shape[0]
        h = anArray.shape[1]
        w = anArray.shape[2]

        sphere = cls.crop_center(anArray, min(d, zz), min(h, yy), min(w,xx))

        c = max((zz - d) // 2, 0)
        cc = max(zz - c - d, 0)

        b = max((yy - h) // 2, 0)
        bb = max(yy - b - h, 0)

        a = max((xx - w) // 2, 0)
        aa = max((xx - a - w), 0)

        result = np.pad(sphere, pad_width=((c, cc), (b, bb), (a, aa)), mode='constant')
        return result


    @classmethod
    def crop_center(cls, img, depth, height, width):
        z, y, x = img.shape
        startZ = z // 2 - (depth // 2)
        startY = y // 2 - (height // 2)
        startX = x // 2 - (width // 2)
        return img[startZ:startZ + depth, startY:startY + height, startX:startX + width]