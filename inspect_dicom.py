import pydicom
from pydicom.errors import InvalidDicomError


import pydicom
from pydicom.errors import InvalidDicomError

def check_dicom_intent(file_path):
    try:
        ds = pydicom.dcmread(file_path)
        
        # Check Presentation Intent Type (0008,0068)
        # Valid values: 'FOR PROCESSING' or 'FOR PRESENTATION'
        intent = ds.get("PresentationIntentType", "Unknown")
        
        print(f"File: {file_path}")
        print(f"Presentation Intent Type: {intent}")
        
        # Additional checks for mammography/advanced imaging
        if "PixelIntensityRelationship" in ds:
            print(f"Pixel Intensity Relationship: {ds.PixelIntensityRelationship}")
            
        return intent
    except InvalidDicomError:
        return "Invalid DICOM File"

# check_dicom_intent(ds)


# trying another method
# checking newfoundland datset
ds = pydicom.dcmread(r"C:\Users\Data Science\Downloads\Attempt2\Deep-LIBRA2\image\IM-0125-0001-0001.dcm")
sop_uid = ds.get("SOPClassUID", None)
print(f"Newfoundland SOP Class UID: {sop_uid}") # still 16 bit raw pixed data

# checking vinndr mammo
ds_vm = pydicom.dcmread(r"C:\Users\Data Science\Downloads\Attempt2\Deep-LIBRA2\image\VinDr Images\T1_L_MLO.dicom")
sop_uid= ds_vm.get("SOPClassUID:", None)
print(f"VinDr SOP Class UID: {sop_uid}")

# TODO breast density ranges, git fork the original repository 