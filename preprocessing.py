import warnings
warnings.filterwarnings("ignore")

# Python packages
import numpy as np
from time import time
from copy import deepcopy
from termcolor import colored
import cv2, os, argparse, pydicom, logging, shutil
from glob import glob
import pydicom

# My packages
from breast_needed_functions import get_headers
from breast_needed_functions import fix_ratio, fix_ratio_to_csv
from breast_needed_functions import Normalize_Image
from breast_needed_functions import object_oriented_preprocessing
from breast_needed_functions import Remove_Top_Below_Side_effect

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output_path", default="./output", help="Path for saving results")
ap.add_argument("-i", "--input_dicom", required=True, help="Folder or single DICOM file")
ap.add_argument("-if", "--image_format", default=".png", help="Image format for saving")
ap.add_argument("-po", "--print_off", type=int, default=0, help="If 1, turns off printing")
ap.add_argument("-ar", "--A_Range", type=int, default=2**8-1, help="The number of bits for saving image")
ap.add_argument("-fis", "--final_image_size", type=int, default=512, help="Final size of image")
ap.add_argument("-sfn", "--saving_folder_name", default=r"air_net_data",
                help="Folder to save preprocessed images for CNN")
ap.add_argument("-lsm", "--libra_segmentation_method", default="Libra", help="Segmentation method")
ap.add_argument("-fpm", "--find_pacemaker", type=int, default=0, help="Remove pacemakers if 1")

args = vars(ap.parse_args())


class Segmentor(object): # The main class
    def __init__(self):
        ######################################################################## Initial
        ######################################################################## Values
        self.Case = args["input_dicom"]
        self.output_path = args["output_path"]
        self.image_format = args["image_format"]
        self.saving_folder_name = args["saving_folder_name"]

        self.A_Range = args["A_Range"]
        self.final_image_size = args["final_image_size"]
        self.print_off = int(args["print_off"])

        self.list_dicom_headers = ["PatientID", "PatientAge", "KVP", "Exposure",
                                   "PresentationIntentType", "Modality", "Manufacturer",
                                   "ImagerPixelSpacing", "BodyPartThickness",
                                   "ImageLaterality", "PhotometricInterpretation",
                                   "Rows", "Columns", "ViewPosition", "FieldOfViewHorizontalFlip"]

        if self.A_Range==2**16-1:
            self.bits_conversion = "uint16"
        elif self.A_Range==2**32-1:
            self.bits_conversion = "uint32"
        else:
            self.bits_conversion = "uint8"

        self.libra_segmentation_method = args["libra_segmentation_method"]
        self.find_pacemaker = args["find_pacemaker"]



    def Main_Loop_Function(self):
        ######################################################################## Couple of
        ######################################################################## initializations
        T_Start = time()


        # no output path = return the results in the same path as dataset
        if self.output_path == '0':
            self.output_path = self.PATH

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # Log file loading
        Path, File = os.path.split(self.Case)
        if File[-4:] == ".dcm": File = File[:-4]
        log_path = os.path.join(self.output_path, "LIBRA_"+File+".log")
        logging.basicConfig(format='%(levelname)s %(asctime)s %(message)s',
                            filename=log_path, level=logging.INFO)
        logging.info('The preprocessing is starting.')


        #################################################################### Loading Image
        #################################################################### & files
        logging.info("The dicom file path is: " +self.Case)
        if self.print_off==0: print(colored("[INFO]", "yellow") + " The dicom file path is: " +
                                    colored(self.Case, "yellow"))

        # dicom folder
        if os.path.isdir(self.Case):
            dicom_files = sorted(glob(os.path.join(self.Case, "*.dicom")) +
                                 glob(os.path.join(self.Case, "*.dcm")))
        else:
            dicom_files = [self.Case]

        if len(dicom_files) == 0:
            raise ValueError("No DICOM files found in input path")

        for dicom_path in dicom_files:
            T_Start = time()
            Path, File = os.path.split(dicom_path)
            File = os.path.splitext(File)[0]

            # load dicoms
            self.ds = pydicom.dcmread(dicom_path)
            self.image = self.ds.pixel_array.astype("float")

            dicom_headers = get_headers(self.ds, self.list_dicom_headers)

            self, self.image_metal = object_oriented_preprocessing(self)
            Remove_Top_Below_Side_effect(self) 
            self.temp_image = deepcopy(self.image)
            print("Mask stats:", self.image.min(), self.image.max(), np.sum(self.image > 0))

        #################################################################### making
        #################################################################### the mask and original image
            logging.info("Saving image")
            if self.print_off==0: print("[INFO] Saving image")


            self.image_16bits = Normalize_Image(self.image, 2**16-1,
                            bits_conversion="uint16", flag_min_edition=True, Min=self.image.min())

            if self.find_pacemaker==1:
                self.image = self.image_metal
            self.image = fix_ratio(self.image,
                                self.final_image_size, self.final_image_size)
            self.image = Normalize_Image(self.image, self.A_Range,
                            bits_conversion=self.bits_conversion, flag_min_edition=True, Min=self.temp_image.min())

            Image_Path = os.path.join(self.output_path, File)
            os.makedirs(Image_Path, exist_ok=True)

            case_folder = os.path.join(self.output_path, File)
            air_mask_folder = os.path.join(case_folder, "air_breast_mask")
            os.makedirs(air_mask_folder, exist_ok=True)

            cv2.imwrite(os.path.join(air_mask_folder, File+"_Normalized"+self.image_format), self.image)
            cv2.imwrite(os.path.join(air_mask_folder, File+"_16bits_Original"+self.image_format), self.image_16bits)
            fix_ratio_to_csv(self.image, self, save_path=air_mask_folder)

            dicom_headers.to_csv(os.path.join(Image_Path, "Headers.csv"))


            Image_Path2 = os.path.join(self.output_path, self.saving_folder_name)
            if not(os.path.isdir(Image_Path2)): os.makedirs(Image_Path2)
            Save_name_img = os.path.join(Image_Path2, File+self.image_format)
            cv2.imwrite(Save_name_img, self.image)


        logging.info("The path of saved image (original normalized image) is: "+Save_name_img)
        if self.print_off==0: print("[INFO] The path of saved image is: "+Save_name_img)


        T_End = time()
        if self.print_off==0: print("[INFO] Elapsed Time (for this file): "+'\033[1m'+ \
              colored(str(round(T_End-T_Start, 2)), 'blue')+'\033[0m'+" seconds")

        logging.info("Preprocessing was successfully done this case.")
        _, new_log_path = os.path.split(log_path)
        new_log_path = os.path.join(Image_Path, new_log_path)
        for handler in logging.root.handlers[:]:
            handler.close()
            logging.root.removeHandler(handler)
        
        shutil.move(log_path, new_log_path)
        if self.print_off==0: print(colored("[INFO]", 'green')+" The breast-air segmentation was successfully processed for this case.")

        logging.info('Segmentation of breast+pectrocal from air (background) starting for ALL CASES using CNN.')


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
############################################################################### Running the code
if __name__ == "__main__":
    Info = Segmentor()
    Info.Main_Loop_Function()

