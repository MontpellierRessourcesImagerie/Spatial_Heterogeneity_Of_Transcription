import numpy as np



class NDArrayUtil:


    @classmethod
    def pad(cls, anArray, xx, yy, zz):
        """
        :param anArray: numpy array
        :param xx: desired height
        :param yy: desired width
        :param zz: desired depth
        :return: padded array
        """

        d = anArray.shape[0]
        h = anArray.shape[1]
        w = anArray.shape[2]

        a = (xx - h) // 2
        aa = xx - a - h

        b = (yy - w) // 2
        bb = yy - b - w

        c = (zz - d) // 2
        cc = zz - c - d

        return np.pad(anArray, pad_width=((a, aa), (b, bb), (c, cc)), mode='constant')