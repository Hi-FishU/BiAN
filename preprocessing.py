import logging
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
from config import Constants
from utils import Model_Logger
import cv2

logger = Model_Logger('SAM')

class InitialMasking(Model_Logger):
    def __init__(self, arg, device):
        super().__init__(arg, device)
        self.arg = arg
        sam = sam_model_registry[Constants.SAM_MODEL_TYPE](Constants.SAM_CKP_PATH)
        sam.to(device)
        self.predictor = SamPredictor(sam)

    def generate_initial_mask(self, img, dot = None) -> np.ndarray:
        if dot is not None:
            contours, _ = cv2.findContours(dot, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            counters = sorted(counters, key=cv2.contourArea, reverse=True)
            dots = []
            for contour in contours:
                if len(contour) != 1:
                    dots.append(contour[0])
                else:
                    dots.append(contour[0])

        self.predictor.set_image(img)
        mask, _, _ = self.predictor.predict(multimask_output=False)
        return mask






