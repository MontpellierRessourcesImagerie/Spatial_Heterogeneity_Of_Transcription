import math
import random
import numpy as np
from scipy.signal import correlate
from skimage.segmentation import clear_border
from skimage.morphology import remove_small_objects
from skimage.segmentation import relabel_sequential
from skimage.measure import regionprops_table
from skimage.measure import regionprops
from bigfish import detection
from scipy.spatial import KDTree, Voronoi
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from scipy.stats import ecdf
from scipy.spatial.distance import cdist
from sphot.measure import TableTool


class Segmentation:
    """ Segment the cells of the embryo and create a label images with the labels 1 to N for the N
    cells in the image that are not touching the borders."""


    def __init__(self, image):
        self.image = image

        self.modelType = "nuclei"
        self.resampleDynamics = False
        self.do3D = True
        self.channels = [0, 0]
        self.diameter = 50
        self.cellProbabilityThreshold = 0
        self.flowThreshold = 0.4
        self.stitchThreshold = 0

        self.clearBorder = True
        self.minSize = 20000

        self.labels = None


    def run(self):
        yield
        self.runCellpose()
        yield
        if self.clearBorder:
            self.labels = clear_border(self.labels)
        yield
        if self.minSize > 0:
            self.labels = remove_small_objects(self.labels, min_size=self.minSize)
        yield
        if self.clearBorder or self.minSize>0:
            self.labels, _, _ = relabel_sequential(self.labels)
        yield


    def runCellpose(self):
        from cellpose import models
        CP = models.CellposeModel(pretrained_model=self.modelType, gpu=True)
        labels, flows_orig, _ = CP.eval(self.image,
                                       channels=self.channels,
                                       diameter=self.diameter,
                                       resample=self.resampleDynamics,
                                       cellprob_threshold=self.cellProbabilityThreshold,
                                       flow_threshold=self.flowThreshold,
                                       do_3D=self.do3D,
                                       z_axis=0,
                                       stitch_threshold=self.stitchThreshold)
        self.labels = labels


    @classmethod
    def keepLabels(cls, labels, labelList):
        newLabels = np.where(np.isin(labels, labelList), labels, 0)
        for index, label in enumerate(labelList, start=1):
            newLabels[newLabels == label] = index
        return newLabels



class SpotDetection:


    def __init__(self, image):
        self.image = image
        self.spots = None
        self.threshold = 4
        self.shallRemoveDuplicates = True
        self.shallFindThreshold = False
        self.scale = (1, 1, 1)
        self.spotRadius = (2.5, 2.5, 2.5)


    def run(self):
        yield
        self.spots = detection.detect_spots(
            self.image,
            threshold=self.threshold,
            remove_duplicate=self.shallRemoveDuplicates,
            return_threshold=self.shallFindThreshold,
            voxel_size=self.scale,
            spot_radius=self.spotRadius)
        self.spots = np.unique(self.spots, axis=0)
        yield



class DecomposeDenseRegions:


    def __init__(self, image, spots):
        self.image = image
        self.spots = spots
        self.voxelSize = (1, 1, 1)
        self.spotRadius = (2.5, 2.5, 2.5)
        self.alpha = 0.5
        self.beta = 1
        self.gamma = 5
        self.referenceSpot = None
        self.decomposedSpots = None


    def run(self):
        yield
        self.decomposedSpots, _, self.referenceSpot = detection.decompose_dense(
                    self.image,
                    self.spots,
                    self.voxelSize,
                    self.spotRadius,
                    alpha=self.alpha, beta=self.beta, gamma=self.gamma
        )
        self.decomposedSpots = np.unique(self.decomposedSpots, axis=0)
        yield



class SpotPerCellAnalyzer:


    def __init__(self, points, labels, scale):
        self.points = points
        self.labels = labels
        self.maxLabel = np.max(labels)
        self.pointsPerCell = {}
        self.nnDistances = {}
        self.allDistances = {}
        self.scale = scale
        self.nnEcdfs = {}
        self.adEcdfs = {}
        self.esEcdfs = {}
        self.nrOfEmptySpacePoints = 1000
        self.emptySpacePointsPerCell = {}
        self.emptySpaceDistances = {}
        self.centroids = {}
        self.distancesFromCentroid = {}


    def getBaseMeasurements(self):
        self._calculateSpotsPerCell()
        props = regionprops_table(self.labels, properties=('label', 'area'),
                                  spacing=self.scale)
        table = {'label': [], 'nucleus_volume': [], 'spots': []}
        for label in range(1, self.maxLabel + 1):
            index = np.where(props['label']==label)[0][0]
            volume = props['area'][index]
            nrOfSpots = len(self.pointsPerCell[label])
            table['label'].append(label)
            table['nucleus_volume'].append(volume)
            table['spots'].append(nrOfSpots)
        return table


    def getNNMeasurements(self):
        self.calculateNNDistances()
        table = {'label': [],
                 'min_nn_dist': [],
                 'mean_nn_dist': [],
                 'std_dev_nn_dist': [],
                 'median_nn_dist': [],
                 'max_nn_dist': []}
        for label in range(1, self.maxLabel+1):
            table['label'].append(label)
            table['min_nn_dist'].append(np.min(self.nnDistances[label][0]))
            table['mean_nn_dist'].append(np.mean(self.nnDistances[label][0]))
            table['std_dev_nn_dist'].append(np.std(self.nnDistances[label][0]))
            table['median_nn_dist'].append(np.median(self.nnDistances[label][0]))
            table['max_nn_dist'].append(np.max(self.nnDistances[label][0]))
        return table


    def getConvexHull(self, label):
        self._calculateSpotsPerCell()
        hull = ConvexHull(self.pointsPerCell[label] / self.scale)
        return hull


    def getDelaunay(self, label):
        self._calculateSpotsPerCell()
        tess = Delaunay(self.pointsPerCell[label] / self.scale)
        return tess


    def getVoronoi(self, label):
        self._calculateSpotsPerCell()
        voro = Voronoi(self.pointsPerCell[label] / self.scale)
        return voro


    def getVoronoiRegions(self, label):
        v = self.getVoronoi(label)
        regions = []
        for i, reg_num in enumerate(v.point_region):
            indices = v.regions[reg_num]
            if -1 in indices:
                continue
            else:
                hull = ConvexHull(v.vertices[indices])
                if not regions:
                    regions = list(hull.points[hull.simplices])
                else:
                    regions = regions + list((hull.points[hull.simplices]))
        return regions


    def getConvexHullMeasurements(self):
        self._calculateSpotsPerCell()
        table = {'label': [],
                 'hull_volume': [],
                 'hull_area': [],
                 'hull_vertices': [],
                 'hull_simplices': [],
                 'bb_depth': [],
                 'bb_height': [],
                 'bb_width': []
                 }
        for label in range(1, self.maxLabel + 1):
            hull = self.getConvexHull(label)
            table['label'].append(label)
            table['hull_volume'].append(hull.volume)
            table['hull_area'].append(hull.area)
            table['hull_vertices'].append(len(hull.vertices))
            table['hull_simplices'].append(len(hull.simplices))
            bounds = hull.max_bound - hull.min_bound
            table['bb_depth'].append(bounds[0])
            table['bb_height'].append(bounds[1])
            table['bb_width'].append(bounds[2])
        return table


    @classmethod
    def tetravol(cls, a, b, c, d):
        '''Calculates the volume of a tetrahedron, given vertices a,b,c and d (triplets)'''
        tetravol=abs(np.dot((a-d),np.cross((b-d),(c-d))))/6
        return tetravol


    def getDelaunayMeasurements(self):
        table = {'label': [],
                 'min_delaunay_vol': [],
                 'mean_delaunay_vol': [],
                 'std_dev_delaunay_vol': [],
                 'median_delaunay_vol': [],
                 'max_delaunay_vol': []}
        for label in range(1, self.maxLabel + 1):
            tess = self.getDelaunay(label)
            volumes = []
            for a, b, c, d in tess.points[tess.simplices]:
                volumes.append(self.tetravol(a, b, c, d))
            volumes = np.array(volumes)
            table['label'].append(label)
            table['min_delaunay_vol'].append(np.min(volumes))
            table['mean_delaunay_vol'].append(np.mean(volumes))
            table['std_dev_delaunay_vol'].append(np.std(volumes))
            table['median_delaunay_vol'].append(np.median(volumes))
            table['max_delaunay_vol'].append(np.max(volumes))
        return table


    def calculateNNDistances(self):
        self._calculateSpotsPerCell()
        self.nnDistances = self.getNNDistances()


    def calculateGFunction(self):
        self._calculateSpotsPerCell()
        self.nnDistances = self.getNNDistances()
        self.calculateNNECDFs()


    def calculateHFunction(self):
        self._calculateSpotsPerCell()
        self.allDistances = self.getAllDistances()
        self.calculateAllECDFs()


    def calculateFFunction(self):
        self._calculateSpotsPerCell()
        self.emptySpaceDistances = self.getEmptySpaceDistances()
        self.calculateEmptySpaceECDFs()


    def calculateDistancesFromCentroid(self):
        self._calculateSpotsPerCell()
        self.distancesFromCentroid = self.getDistancesFromCentroid()
        self.calculateCentroidECDFs()


    def _calculateSpotsPerCell(self):
        if self.pointsPerCell:
            return
        maxLabel = int(np.max(self.labels))
        for i in range(0, maxLabel + 1):
            self.pointsPerCell[i] = []
        for point in self.points:
            label = self.labels[int(point[0]), int(point[1]), int(point[2])]
            self.pointsPerCell[label].append(point * self.scale)


    def getNNDistances(self):
        nnDistances = {}
        for label in range(1, self.maxLabel + 1):
            data = self.pointsPerCell[label]
            nnDistances[label] = self.getNNDistancesFor(data)
        return nnDistances


    def getNNDistancesFor(self, data):
        kdtree = KDTree(data)
        dist, points = kdtree.query(data, 2)
        nnDistances = (dist[:,1], points)
        return nnDistances


    def getDistancesFromCentroid(self):
        props = regionprops(self.labels)
        for label in range(1, self.maxLabel + 1):
            self.centroids[label] = props[label].centroid
            data = self.pointsPerCell[label]
            self.distancesFromCentroid[label] = self.getDistancesFromCentroidFor(data, self.centroids[label])
        return self.distancesFromCentroid


    def getEmptySpaceDistances(self):
        emptySpaceDistances = {}
        for label in range(1, self.maxLabel + 1):
            data = self.pointsPerCell[label]
            self.emptySpacePointsPerCell[label] = self.getNRandomPointsForLabel(label, self.nrOfEmptySpacePoints)
            emptySpaceDistances[label] = self.getEmptySpaceDistancesFor(data, self.emptySpacePointsPerCell[label])
        return emptySpaceDistances


    def getEmptySpaceDistancesFor(self, data, refPoints):
        kdtree = KDTree(data)
        dist, points = kdtree.query(refPoints, 2)
        nnDistances = (dist[:, 1], points)
        return nnDistances


    def getDistancesFromCentroidFor(self, data, centroid):
        distances = []
        for point in data:
            distances.append(np.linalg.norm(np.array(point) - np.array(centroid)))
        distances.sort()
        return distances


    def getAllDistances(self):
        allDistances = {}
        for label in range(1, self.maxLabel + 1):
            data = self.pointsPerCell[label]
            allDistances[label] = self.getAllDistancesFor(data)
        return allDistances


    def getAllDistancesFor(self, data):
        N = len(data)
        dist = cdist(data, data, 'euclidean')
        allDistances = (dist[np.triu_indices(N, 1)], data)
        return allDistances


    def calculateNNECDFs(self):
        self.nnEcdfs = {}
        for label in range(1, self.maxLabel + 1):
            self.nnEcdfs[label] = self.getECDF(self.nnDistances[label][0])


    def calculateAllECDFs(self):
        self.adEcdfs = {}
        for label in range(1, self.maxLabel + 1):
            self.adEcdfs[label] = self.getECDF(self.allDistances[label][0])


    def calculateEmptySpaceECDFs(self):
        self.esEcdfs = {}
        for label in range(1, self.maxLabel + 1):
            self.esEcdfs[label] = self.getECDF(self.emptySpaceDistances[label][0])


    def getRandomPointsForLabel(self, label):
        nrOfPoints = len(self.pointsPerCell[label])
        return self.getNRandomPointsForLabel(label, nrOfPoints)


    def getNRandomPointsForLabel(self, label, N):
        labelsForLabel = np.ndarray.copy(self.labels)
        labelsForLabel[labelsForLabel < label] = 0
        labelsForLabel[labelsForLabel > label] = 0
        nz = np.nonzero(labelsForLabel)
        coords = np.transpose(nz) * self.scale
        maxSample = len(coords) - 1
        indices = random.sample(range(maxSample + 1), N)
        randPoints = coords[indices]
        return randPoints


    @classmethod
    def getECDF(cls, distances):
        res = ecdf(distances)
        return res


    def getEnvelopForNNDistances(self, label, nrOfSamples=100):
        maxDist = np.max(self.nnDistances[label][0])
        return self.getEnvelopForDistanceFunction(label, self.getNNDistancesFor, maxDist, nrOfSamples=nrOfSamples)
        # return self.getEnvelopFromECDFsOrdering(label, self.getNNDistancesFor, maxDist, nrOfSamples=nrOfSamples)


    def getEnvelopForAllDistances(self, label, nrOfSamples=100):
        maxDist = np.max(self.allDistances[label][0])
        return self.getEnvelopForDistanceFunction(label, self.getAllDistancesFor, maxDist, nrOfSamples=nrOfSamples)
        # return self.getEnvelopFromECDFsOrdering(label, self.getAllDistancesFor, maxDist, nrOfSamples=nrOfSamples)


    def getEnvelopForEmptySpaceDistances(self, label, nrOfSamples=100):
        maxDist = np.max(self.emptySpaceDistances[label][0])
        lower95thIndex = (5 * nrOfSamples) // 100
        upper95thIndex = (95 * nrOfSamples) // 100
        xValues = np.array(list(np.arange(0, math.floor(maxDist + 1), self.scale[1])))
        envelops = np.zeros((nrOfSamples, len(xValues)))
        for i in range(nrOfSamples):
            points = self.getRandomPointsForLabel(label)
            distances = self.getEmptySpaceDistancesFor(points, self.emptySpacePointsPerCell[label])[0]
            cdf = ecdf(distances)
            row = cdf.cdf.evaluate(xValues)
            row = row.tolist()
            for index, element in enumerate(row):
                envelops[i, index] = element
        maxEnvs = np.zeros(len(xValues))
        minEnvs = np.zeros(len(xValues))
        upper95ths = np.zeros(len(xValues))
        lower95ths = np.zeros(len(xValues))
        for i in range(len(xValues)):
            column = np.sort(envelops[:, i])
            minEnvs[i] = column[0]
            lower95ths[i] = column[lower95thIndex]
            upper95ths[i] = column[upper95thIndex]
            maxEnvs[i] = column[nrOfSamples - 1]
        return minEnvs, lower95ths, upper95ths, maxEnvs


    def getEnvelopForDistanceFunction(self, label, distanceFunction, maxDist, nrOfSamples=100):
        lower95thIndex = (5 * nrOfSamples) // 100
        upper95thIndex = (95 * nrOfSamples) // 100
        xValues = np.array(list(np.arange(0, math.floor(maxDist+1), self.scale[1])))
        envelops = np.zeros((nrOfSamples, len(xValues)))
        for i in range(nrOfSamples):
            points = self.getRandomPointsForLabel(label)
            distances = distanceFunction(points)[0]
            cdf = ecdf(distances)
            row = cdf.cdf.evaluate(xValues)
            row = row.tolist()
            for index, element in enumerate(row):
                envelops[i, index] = element
        maxEnvs = np.zeros(len(xValues))
        minEnvs = np.zeros(len(xValues))
        upper95ths = np.zeros(len(xValues))
        lower95ths = np.zeros(len(xValues))
        for i in range(len(xValues)):
            column = np.sort(envelops[:, i])
            minEnvs[i] = column[0]
            lower95ths[i] = column[lower95thIndex]
            upper95ths[i] = column[upper95thIndex]
            maxEnvs[i] = column[nrOfSamples-1]
        return minEnvs, lower95ths, upper95ths, maxEnvs


    def getEnvelopFromECDFsOrdering(self, label, distanceFunction, maxDist, nrOfSamples=100):
        lower95thIndex = (5 * nrOfSamples) // 100
        upper95thIndex = (95 * nrOfSamples) // 100
        scoredECDFs = []
        for i in range(nrOfSamples):
            points = self.getRandomPointsForLabel(label)
            distances = distanceFunction(points)[0]
            scoredECDF = ScoredECDF(ecdf(distances), maxDist, self.scale[1])
            scoredECDFs.append(scoredECDF)
        scoredECDFs.sort(key = lambda x: x.score)
        minEnv = scoredECDFs[0].yValues
        lower95th = scoredECDFs[lower95thIndex].yValues
        upper95th = scoredECDFs[upper95thIndex].yValues
        maxEnv = scoredECDFs[nrOfSamples-1].yValues
        return minEnv, lower95th, upper95th, maxEnv


    @classmethod
    def getPointImageFor(cls, points):
        pointsT = np.transpose(points)
        minZ = np.min(pointsT[0])
        maxZ = np.max(pointsT[0])
        minY = np.min(pointsT[1])
        maxY = np.max(pointsT[1])
        minX = np.min(pointsT[2])
        maxX = np.max(pointsT[2])
        depth = (maxZ - minZ) + 1
        height = (maxY - minY) + 1
        width = (maxX - minX) + 1
        image = np.zeros((depth, height, width), np.uint8)
        shiftedPoints = [np.array([z-minZ, y-minY, x-minX]) for z, y, x in points]
        for z, y, x in shiftedPoints:
            image[z][y][x] = 255
        return image


    def cropImageForLabel(self, image, label):
        """
        Crop the image to the bounding box of the given label and set the pixels outside the label
        to zero.
        """
        props = regionprops(self.labels)
        bbox = props[label-1].bbox
        result = image[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]]
        mask = self.labels[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]]
        mask = (mask == label) * 1
        result = result * mask
        result = Background.removeMin(result)
        return result



class ScoredECDF:


    def __init__(self, cdf, maxDist, scale):
        self.cdf = cdf
        self.maxDist = maxDist
        self.scale = scale
        self.score = 0
        self.yValues = []
        self._calculateScore()


    def _calculateScore(self):
        xValues = np.array(list(range(0, math.floor(self.maxDist + 1), self.scale)))
        self.yValues = self.cdf.cdf.evaluate(xValues).tolist()
        self.score = sum(self.yValues)



class Background(object):


    def __init__(self):
        super(Background, self).__init__()


    @classmethod
    def removeMin(cls, image):
        val = np.min(image[image>0])
        image = np.clip(image-val, 0, np.max(image))
        return image



class Coordinates(object):


    def __init__(self, origin):
        super(Coordinates, self).__init__()
        self.origin = origin


    @staticmethod
    def cartesianToPolar(y, x):
        radius = math.sqrt(x * x + y * y)
        theta = math.atan2(y, x)
        return radius, theta


    @staticmethod
    def polarToCartesian(radius, theta):
        y = radius * math.sin(theta)
        x = radius * math.cos(theta)
        return y, x


    def cartesianToSpherical(self, z, y, x):
        z = z + self.origin[0]
        y = y + self.origin[1]
        x = x + self.origin[2]
        xy, phi = Coordinates.cartesianToPolar(y, x)
        radius, theta = Coordinates.cartesianToPolar(z, xy)
        return radius, theta, phi


    def sphericalToCartesian(self, radius, theta, phi):
        xy, z = Coordinates.polarToCartesian(radius, theta)
        y, x = Coordinates.polarToCartesian(xy, phi)
        z = z + self.origin[0]
        y = y + self.origin[1]
        x = x + self.origin[2]
        return z, y, x


    @staticmethod
    def getSphereRanges(maxRadius):
        radii = np.arange(0, maxRadius, 1).tolist()
        inclinations = np.arange(-np.pi, np.pi, np.pi/180).tolist()
        azimuths = np.arange(-np.pi, np.pi, np.pi/180).tolist()
        return radii, inclinations, azimuths



class Correlator(object):


    def __init__(self, image1, image2=None):
        super(Correlator, self).__init__()
        self.image1 = image1
        self.image2 = image2
        self.correlationImage = None
        self.correlationProfile = None
        self.usePadding = True
        self.paddingMode = 'wrap'


    def reset(self):
        self.correlationImage = None
        self.correlationProfile = None


    def calculateAutoCorrelation(self):
        image = self.image1
        image1 = image
        if self.usePadding:
            image1 = self.pad(image)
        self.correlationImage = correlate(image1, image, mode="valid")


    def calculateCrossCorrelation(self):
        image1 = self.image1.astype(float)
        image2 = self.image2.astype(float)
        norm1 = np.linalg.norm(image1)
        image1 = image1 / norm1
        norm2 = np.linalg.norm(image2)
        image2 = image2 / norm2
        if self.usePadding:
            image1 = self.pad(image1)
        self.correlationImage = correlate(image1, image2,  mode="valid")


    def pad(self, image):
        paddedImage = np.pad(image,
                             [(image.shape[0], ), (image.shape[1],), (image.shape[2],)],
                             mode=self.paddingMode)
        return paddedImage


    def calculateAutoCorrelationProfile(self):
        if self.correlationImage is None:
            self.calculateAutoCorrelation()
        self.calculateRadialCorrelationProfile()


    def calculateCrossCorrelationProfile(self):
        if self.correlationImage is None:
            self.calculateCrossCorrelation()
        self.calculateRadialCorrelationProfile()


    def calculateRadialCorrelationProfile(self):
        maxRadius = min(self.correlationImage.shape) // 2
        originZ, originY, originX = (self.correlationImage.shape[0] // 2,
                                     self.correlationImage.shape[1] // 2,
                                     self.correlationImage.shape[2] // 2)
        coords = Coordinates((originZ, originY, originX))
        radii, inclinations, azimuths = Coordinates.getSphereRanges(maxRadius)
        meanByRadius = [0] * len(radii)
        N = len(inclinations) * len(azimuths)
        for i, radius in enumerate(radii):
            for inclination in inclinations:
                for azimuth in azimuths:
                    z, y, x = coords.sphericalToCartesian(radius, inclination, azimuth)
                    z ,y, x = round(z), round(y), round(x)
                    meanByRadius[i] = meanByRadius[i] + self.correlationImage[z, y, x]
            meanByRadius[i] = meanByRadius[i] / N
        self.correlationProfile = (radii, meanByRadius)


    def drawSphere(self, radius):
        originZ, originY, originX = (self.correlationImage.shape[0] // 2,
                                     self.correlationImage.shape[1] // 2,
                                     self.correlationImage.shape[2] // 2)
        coords = Coordinates((originZ, originY, originX))
        _, inclinations, azimuths = Coordinates.getSphereRanges(radius)
        surfaceImage = np.zeros(self.correlationImage.shape)
        for inclination in inclinations:
            for azimuth in azimuths:
                z, y, x = coords.sphericalToCartesian(radius, inclination, azimuth)
                z, y, x = round(z), round(y), round(x)
                surfaceImage[z, y, x] = 255
        return surfaceImage



class SpatialStatFunction(object):


    def __init__(self, spots, labels, scale, unit, label):
        super().__init__()
        self.nrOfSamples = 100
        self.spots = spots
        self.labels = labels
        self.scale = scale
        self.unit = unit
        self.label = label
        self.envelop = None
        self.analyzer = SpotPerCellAnalyzer(self.spots, self.labels, self.scale)


    def run(self):
        self.subclassResponsability()



class FFunctionTask(SpatialStatFunction):


    def __init__(self, spots, labels, scale, unit, label):
        super(FFunctionTask, self).__init__(spots, labels, scale, unit, label)


    def run(self):
        self.analyzer.calculateFFunction()
        self.envelop = self.analyzer.getEnvelopForEmptySpaceDistances(self.label, nrOfSamples=self.nrOfSamples)



class GFunctionTask(SpatialStatFunction):


    def __init__(self, spots, labels, scale, unit, label):
        super(GFunctionTask, self).__init__(spots, labels, scale, unit, label)


    def run(self):
        self.analyzer.calculateGFunction()
        self.envelop = self.analyzer.getEnvelopForNNDistances(self.label, nrOfSamples=self.nrOfSamples)



class HFunctionTask(SpatialStatFunction):


    def __init__(self, spots, labels, scale, unit, label):
        super(HFunctionTask, self).__init__(spots, labels, scale, unit, label)


    def run(self):
        self.analyzer.calculateHFunction()
        self.envelop = self.analyzer.getEnvelopForAllDistances(self.label, nrOfSamples=self.nrOfSamples)



class TesselationTask(object):


    def __init__(self, spots, labels, scale, unit, label):
        super().__init__()
        self.spots = spots
        self.labels = labels
        self.scale = scale
        self.unit = unit
        self.label = label
        self.result = None
        self.analyzer = SpotPerCellAnalyzer(self.spots, self.labels, self.scale)


    def run(self):
        self.subclassResponsability()



class ConvexHullTask(TesselationTask):


    def __init__(self, spots, labels, scale, unit, label):
        super(ConvexHullTask, self).__init__(spots, labels, scale, unit, label)


    def run(self):
        analyzer = SpotPerCellAnalyzer(self.spots, self.labels, self.scale)
        self.result = analyzer.getConvexHull(self.label)



class DelaunayTask(TesselationTask):


    def __init__(self, spots, labels, scale, unit, label):
        super(DelaunayTask, self).__init__(spots, labels, scale, unit, label)


    def run(self):
        analyzer = SpotPerCellAnalyzer(self.spots, self.labels, self.scale)
        self.result = analyzer.getDelaunay(self.label)


class VoronoiTask(TesselationTask):


    def __init__(self, spots, labels, scale, unit, label):
        super(VoronoiTask, self).__init__(spots, labels, scale, unit, label)


    def run(self):
        analyzer = SpotPerCellAnalyzer(self.spots, self.labels, self.scale)
        self.result = analyzer.getVoronoiRegions(self.label)



class MeasureTask:


    def __init__(self, spots, labels, scale, units):
        self.spots = spots
        self.labels = labels
        self.scale = scale
        self.units = units
        self.analyzer = SpotPerCellAnalyzer(spots, labels, scale)
        self.table = None


    def run(self):
        baseMeasurements = self.analyzer.getBaseMeasurements()
        nnMeasurements = self.analyzer.getNNMeasurements()
        nnMeasurements.pop('label')
        TableTool.addColumnsTableAToB(nnMeasurements, baseMeasurements)
        convexHullMeasurements = self.analyzer.getConvexHullMeasurements()
        convexHullMeasurements.pop('label')
        TableTool.addColumnsTableAToB(convexHullMeasurements, baseMeasurements)
        delaunayMeasurements = self.analyzer.getDelaunayMeasurements()
        delaunayMeasurements.pop('label')
        TableTool.addColumnsTableAToB(delaunayMeasurements, baseMeasurements)
        self.table = baseMeasurements
        self.table['base unit'] = [self.units[0]]*len(self.table['label'])



class CropLabelTask:


    def __init__(self, labels, image, label):
        self.labels = labels
        self.image = image
        self.label = label
        self.result = None


    def run(self):
        analyzer = SpotPerCellAnalyzer(None, self.labels, 1)
        self.result = analyzer.cropImageForLabel(self.image, self.label)
