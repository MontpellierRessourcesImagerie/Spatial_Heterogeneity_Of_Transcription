from skimage.segmentation import clear_border
from skimage.morphology import remove_small_objects
from skimage.segmentation import relabel_sequential



class Segmentation:
    ''' Segment the cells of the embryon and create a label images with the labels 1 to N for the N
    cells in the image that are not touching the borders.
    '''


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
        self.runCellpose()
        print(self.labels.shape)
        if self.clearBorder:
            self.labels = clear_border(self.labels)
        print(self.labels.shape)
        if self.minSize > 0:
            self.labels = remove_small_objects(self.labels, min_size=self.minSize)
        print(self.labels.shape)
        if self.clearBorder or self.minSize>0:
            self.labels, _, _ = relabel_sequential(self.labels)
        print(self.labels.shape)


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

