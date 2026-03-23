import numpy as np
import cv2
from PIL import Image
import pydicom


def dicom_to_image(dicom_path,output_path,format="png"):
    try:
        dicom=pydicom.dcmread(dicom_path)
        pixel_array = dicom.pixel_array
        # if 'PhotometricInterpretation' not in dicom:
        #     raise RuntimeError(f"Missing 'PhotometricInterpretation' attribute in DICOM file: {dicom_path}")
        pixel_array=(pixel_array-pixel_array.min())/(pixel_array.max()-pixel_array.min())*255
        pixel_array=pixel_array.astype(np.uint8)
        print(f"shape of pixel array {pixel_array.shape}")
        if dicom.PhotometricInterpretation=="MONOCHROME1":
            pixel_array=255-pixel_array

        if format=="jpg":
            cv2.imwrite(output_path,pixel_array)
        else:
            Image.fromarray(pixel_array).save(output_path)

    except Exception as e:
        raise RuntimeError(f"Failed to convert Dicom to {format}:{str(e)}")