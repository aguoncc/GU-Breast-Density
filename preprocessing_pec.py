import warnings
warnings.filterwarnings("ignore")

# Python packages
import numpy as np
from time import time
from skimage import exposure
from termcolor import colored
import cv2, os, argparse, logging
from glob import glob

# My packages
from breast_needed_functions import find_logical_background_objs
from breast_needed_functions import Normalize_Image, find_largest_obj


################################## This script is for training the svm
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output_path", default='./output',
                help="path for saving results file")
ap.add_argument("-i", "--input", required=False, default='Full_path_to_image_name',
                help="path for input file or folder")
ap.add_argument("-if", "--image_format", default='.png', help="Image format for saving")
ap.add_argument("-po", "--print_off", type=int, default=0, help="If 1, turn off printing")
ap.add_argument("-ar", "--A_Range", type=int, default=2**8-1, help="Number of bits for saving image")
ap.add_argument("-fis", "--final_image_size", type=int, default=512, help="Final size of image")
ap.add_argument("-sfn", "--saving_folder_name", default="pec_net_data/image",
                help="Folder for saving preprocessed images")
ap.add_argument("-cn", "--case_name", default=None,
                help="Optional: If single file, case name; otherwise inferred from file name")

args = vars(ap.parse_args())


class Segmentor(object): # the main class
    ######################################################################## Initial
    ######################################################################## Values
    def __init__(self):
        self.input_path = args["input"]
        self.image_format = args["image_format"]
        self.saving_folder_name = args["saving_folder_name"]
        self.output_path = args["output_path"]
        self.A_Range = args["A_Range"]
        self.final_image_size = args["final_image_size"]
        self.print_off = int(args["print_off"])
        self.threshold_percentile = 0.5

        if self.A_Range==2**16-1:
            self.bits_conversion = "uint16"
        elif self.A_Range==2**32-1:
            self.bits_conversion = "uint32"
        else:
            self.bits_conversion = "uint8"

    def Main_Loop_Function(self):
        ######################################################################## Couple of
        ######################################################################## initializations
        if os.path.isdir(self.input_path):
            # Windows-compatible: grab all PNGs in folder
            image_files = sorted(glob(os.path.join(self.input_path, "*.png")))
            if len(image_files) == 0:
                raise ValueError(f"No images found in folder: {self.input_path}")
        else:
            image_files = [self.input_path]

        for img_file in image_files:
            T_Start = time()
            case_name = args["case_name"] if args["case_name"] else os.path.splitext(os.path.basename(img_file))[0]

            if not self.print_off:
                print(colored("[INFO]", "yellow"), "Processing:", colored(case_name, "yellow"))

            # load original air-breast mask
            org_image_path = os.path.join(self.output_path, case_name,
                                          "air_breast_mask", case_name + "_Normalized" + self.image_format)
            self.org_image = cv2.imread(org_image_path, -1)
            if self.org_image is None:
                raise FileNotFoundError(f"Original normalized image not found: {org_image_path}")

            # load mask (mask path can also be folder input)
            self.mask = cv2.imread(img_file, 0)
            self.mask = self.mask > 0

            try:
                self.mask = find_logical_background_objs(self.mask)
                self.mask = find_largest_obj(self.mask)
            except:
                if not self.print_off:
                    print("Warning: Mask issue for case:", case_name)
                logging.info("The breast air CNN mask had an issue. Using largest object fallback.")
                self.mask = find_largest_obj(self.mask)

            # apply mask to original image
            self.org_image[np.logical_not(self.mask)] = 0
            try:
                Min = self.org_image[self.mask].min()
            except:
                Min = self.org_image.min()
            self.org_image[np.logical_not(self.mask)] = Min

            # normalize and scale
            non_zero = self.org_image[self.org_image>Min]
            if len(non_zero) > 0:
                self.image = (self.org_image - (np.percentile(non_zero, self.threshold_percentile) - 1/self.A_Range)) / \
                             (non_zero.max() - (np.percentile(non_zero, self.threshold_percentile) - 1/self.A_Range))
            else:
                self.image = self.org_image / self.org_image.max()

            self.image[self.image<0] = 0
            self.image = (self.image * self.A_Range).astype(self.bits_conversion)

            self.image_he = exposure.equalize_hist(self.image, nbins=self.A_Range, mask=self.mask>0)
            self.image_he[self.mask==0] = 0
            self.image_he = Normalize_Image(self.image_he, self.A_Range-1, bits_conversion=self.bits_conversion)
            self.image_he += 1
            self.image_he[self.mask==0] = 0

            self.org_image[self.mask==0] = 0
            self.image_main = Normalize_Image(self.org_image, self.A_Range-1, bits_conversion=self.bits_conversion)
            self.image_main += 1
            self.image_main[self.mask==0] = 0

            self.image = np.concatenate((
                self.image.reshape([self.final_image_size, self.final_image_size, 1]),
                self.image_he.reshape([self.final_image_size, self.final_image_size, 1])
            ), axis=2)
            self.image = np.concatenate((
                self.image,
                self.image_main.reshape([self.final_image_size, self.final_image_size, 1])
            ), axis=2).astype(self.bits_conversion)

        #################################################################### saving images
        #################################################################### in folders
            base_case_path = os.path.join(self.output_path, case_name)
            os.makedirs(base_case_path, exist_ok=True)

            mask_path = os.path.join(base_case_path, "air_breast_mask")
            os.makedirs(mask_path, exist_ok=True)
            cv2.imwrite(os.path.join(mask_path, case_name + "_air_breast_mask" + self.image_format),
                        self.mask.astype(self.bits_conversion) * self.A_Range)

            breast_mask_path = os.path.join(base_case_path, "breast_mask")
            os.makedirs(breast_mask_path, exist_ok=True)
            cv2.imwrite(os.path.join(breast_mask_path, case_name + "_pec_breast_preprocessed" + self.image_format),
                        self.image)

            cnn_path = os.path.join(self.output_path, self.saving_folder_name)
            os.makedirs(cnn_path, exist_ok=True)
            cv2.imwrite(os.path.join(cnn_path, case_name + self.image_format), self.image)

            if not self.print_off:
                print(colored("[INFO]", "green"), f"Saved preprocessed case: {case_name} in {round(time()-T_Start,2)}s")


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
############################################################################### Running the code
if __name__ == "__main__":
    Segmentor().Main_Loop_Function()
