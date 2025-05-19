import math
import random
import numpy as np
from skimage.segmentation import clear_border
from skimage.morphology import remove_small_objects
from skimage.segmentation import relabel_sequential
from bigfish import detection
from scipy.spatial import KDTree
from scipy.stats import ecdf
from scipy.spatial.distance import cdist



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



class SpotDetection:


    def __init__(self, image):
        self.image = image
        self.spots = None
        self.threshold = 4
        self.shallRemoveDuplicates = True
        self.shallFindThreshold = False
        self.scale = (50, 50, 50)
        self.spotRadius = 150


    def run(self):
        yield
        self.spots = detection.detect_spots(
            self.image,
            threshold=self.threshold,
            remove_duplicate=self.shallRemoveDuplicates,
            return_threshold=self.shallFindThreshold,
            voxel_size=self.scale,
            spot_radius=self.spotRadius)
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


    def getNNMeasurements(self):
        self.calculateNNDistances()
        table = {'label': [],
                 'min-nn-dist': [],
                 'mean-nn-dist': [],
                 'std-dev-nn-dist': [],
                 'median-nn-dist': [],
                 'max-nn-dist': []}
        for label in range(1, self.maxLabel+1):
            table['label'].append(label)
            table['min-nn-dist'].append(np.min(self.nnDistances[label][0]))
            table['mean-nn-dist'].append(np.mean(self.nnDistances[label][0]))
            table['std-dev-nn-dist'].append(np.std(self.nnDistances[label][0]))
            table['median-nn-dist'].append(np.median(self.nnDistances[label][0]))
            table['max-nn-dist'].append(np.max(self.nnDistances[label][0]))
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


    def _calculateSpotsPerCell(self):
        maxLabel = int(np.max(self.labels))
        for i in range(0, maxLabel + 1):
            self.pointsPerCell[i] = []
        for point in self.points:
            label = self.labels[point[0], point[1], point[2]]
            self.pointsPerCell[label].append(point)


    def getNNDistances(self):
        nnDistances = {}
        for label in range(1, self.maxLabel + 1):
            data = self.pointsPerCell[label]
            nnDistances[label] = self.getNNDistancesFor(data)
        return nnDistances


    def getNNDistancesFor(self, data):
        kdtree = KDTree(data)
        dist, points = kdtree.query(data, 2)
        nnDistances = (dist[:,1] * self.scale, points)
        return nnDistances


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
        nnDistances = (dist[:, 1] * self.scale, points)
        return nnDistances


    def getAllDistances(self):
        allDistances = {}
        for label in range(1, self.maxLabel + 1):
            data = self.pointsPerCell[label]
            allDistances[label] = self.getAllDistancesFor(data)
        return allDistances


    def getAllDistancesFor(self, data):
        N = len(data)
        dist = cdist(data, data, 'euclidean')
        allDistances = (dist[np.triu_indices(N, 1)] * self.scale, data)
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
        coords = np.transpose(nz)
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
        return self.getEnvelopForDistanceFunction(label, self.getNNDistancesFor, maxDist, nrOfSamples)


    def getEnvelopForAllDistances(self, label, nrOfSamples=100):
        maxDist = np.max(self.allDistances[label][0])
        return self.getEnvelopForDistanceFunction(label, self.getAllDistancesFor, maxDist, nrOfSamples)


    def getEnvelopForEmptySpaceDistances(self, label, nrOfSamples=100):
        maxDist = np.max(self.emptySpaceDistances[label][0])
        lower95thIndex = (5 * nrOfSamples) // 100
        upper95thIndex = (95 * nrOfSamples) // 100
        xValues = np.array(list(range(0, math.floor(maxDist + 1), self.scale)))
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
        xValues = np.array(list(range(0, math.floor(maxDist+1), self.scale)))
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
