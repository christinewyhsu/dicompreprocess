"""Parsing code for DICOMS and contour files"""

import pydicom
from pydicom.errors import InvalidDicomError

import numpy as np
from PIL import Image, ImageDraw

import logging

def _parse_contour_file(filename):
    """
    This method parses the given contour filename and also
    checks whether the contour file contains at least three coordinates
    
    Args:
       filename (str): filepath to the contourfile to parse

    Return:
       list of tuples holding x, y coordinates of the contour
    """

    coords_lst = []

    with open(filename, 'r') as infile:
        for line in infile:
            coords = line.strip().split()

            x_coord = float(coords[0])
            y_coord = float(coords[1])
            coords_lst.append((x_coord, y_coord))
     
     # Contour file must contain at minimum three coordinates to draw the simplest polygon 
    if len(coords_lst) < 3:
        raise Exception('''Contour file {} is invalid as it 
            contains less than three coordinates'''.format(filename.split('/')[-1]))

    return coords_lst


def _parse_dicom_file(filename):
    """Parse the given DICOM filename

    Args:
       filename (str): filepath to the DICOM file to parse
    Return:
       DICOM image data array
    """

    try:
        dcm = pydicom.read_file(filename)
        dcm_image = dcm.pixel_array

        try:
            intercept = dcm.RescaleIntercept
        except AttributeError:
            intercept = 0.0
        try:
            slope = dcm.RescaleSlope
        except AttributeError:
            slope = 0.0

        if intercept != 0.0 and slope != 0.0:
            dcm_image = dcm_image*slope + intercept           
        return dcm_image
    except InvalidDicomError:
        return None


def _poly_to_mask(polygon, width, height):
    """Convert polygon to mask
    
    Args:
       polygon: list of pairs of x, y coords [(x1, y1), (x2, y2), ...] in units of pixels
       width: scalar image width
       height: scalar image height
    Return:
       Boolean mask of shape (height, width)
    """

    # http://stackoverflow.com/a/3732128/1410871
    img = Image.new(mode='L', size=(width, height), color=0)
    ImageDraw.Draw(img).polygon(xy=polygon, outline=0, fill=1)
    mask = np.array(img).astype(bool)
    return mask
