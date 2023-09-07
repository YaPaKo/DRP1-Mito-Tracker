import getopt
import os
import re
import sys
from math import ceil

import cupy as cp
import pandas as pd
from PIL import Image
from aicsimageio import AICSImage
from numpy import zeros, matrix, uint32, ndarray
from skimage.filters import threshold_li, threshold_otsu

from tracking import tracker
from mito_skel import video_skeleton_data, skeleton_as_dataframe
from tubular_projection import add_mito_data

FILE_NAME_SUFFIX = "tif"


def extract(dir_name):
    """
    Extracts the trajectories of the drp1, the skeleton of the mitochondria
    and put them together. Saves the trajectories as the same file name with
    '_tracking.csv' as ending and the mitochondrial skeleton with the
    '_mitoskel.csv' ending.

    Note: Currently only up to 4GB tif files are supported.

    :param dir_name: The directory in which the videos are
    """
    files = []
    for (_, _, file_names) in os.walk(dir_name):
        for f in file_names:
            if f.endswith(f".{FILE_NAME_SUFFIX}"):
                files.append(f)
        break

    if len(files) == 0:
        print("No files found for", dir_name)
        return None

    # Extracts every tif video in the given folder
    for file_name in files:
        base_filename = file_name[:-4]
        path = os.path.join(dir_name, base_filename + "." + FILE_NAME_SUFFIX)

        # Get video from the path
        video = tif_to_array(path)

        # Splitting the video in the two channels
        mito = video[0::2]
        drp1 = video[1::2]

        # DRP1 tracking
        print("Collecting DRP1 tracking data")
        drp1_df = tracker(cp.array(drp1))

        # Thresholds for the mitochondria extractions
        th = threshold_li(mito)
        mth = threshold_otsu(mito)

        # Number of chunks for the mitochondrial skeleton extraction. Too much is problematic for Windows memory.
        print("Collecting mitochondrial skeleton data")
        c = 64
        count = ceil(len(mito) / c)
        res = []
        # Extract mitochondrial skeletons
        for i in range(count):
            res.append(skeleton_as_dataframe(video_skeleton_data(mito[i * c:min(len(mito), (i + 1) * c)], th, mth)))
            res[i].t += i * c
        mito_df = pd.concat(res)
        mito_df.index = range(len(mito_df))

        # Free memory
        del video, mito, drp1, th, mth

        # Project the DRP1 trajectories to the tubular mitochondrial projection
        print("Projecting tracking data to mitochondrial skeleton")
        drp1_df = add_mito_data(drp1_df, mito_df)

        # Preparing path
        tracking_path = os.path.join(dir_name, base_filename + "_tracking" + '.csv')
        mitoskel_path = os.path.join(dir_name, base_filename + "_mitoskel" + '.csv')

        # Saving data as csv files
        print("Saving csv files")
        drp1_df.to_csv(tracking_path, index=False)
        mito_df.to_csv(mitoskel_path, index=False)


def tif_to_array(path: str, t_start=0, t_end=-1, dtype=uint32) -> (ndarray, float, float, str):
    """ Loading the stack from path into numpy array with dimension (t,y,x)

    :param path: Path to the image
    :param t_start: First frame of the image series (optional)
    :param t_end: Last frame of the image series (optional)
    :param dtype: Data type for the return value
    :return: Numpy array with dimension (t,y,x)
    """
    with Image.open(path) as im:
        if t_end == -1:
            t_end = im.n_frames
        im_stack = zeros((t_end - t_start, im.height, im.width), dtype=dtype)
        i = 0
        for t in range(t_start, t_end):
            im.seek(t)
            # noinspection PyTypeChecker
            im_stack[i] = matrix(im, dtype=dtype)
            i = i + 1

    return im_stack


def czi_to_array(path: str, t_start=0, t_end=-1, channels=None):
    """ Loading the stack from path into numpy array with dimension (t,y,x)

    :param path: Path to the image
    :param t_start: First frame of the image series (optional)
    :param t_end: Last frame of the image series (optional)
    :param channels: Which channels to return {mito, drp1, mtdna}
    :return: Numpy array with dimension (t,y,x)
    """
    # Should be (T,C,Z,Y,X)
    im = AICSImage(path)

    # Use full length if no end frame was provided
    if t_end == -1:
        t_end = im.shape[0]

    data = {}
    for i, channel in enumerate(
            im.metadata.find('Metadata')
                       .find('Information')
                       .find('Image')
                       .find('Dimensions')
                       .find('Channels')
    ):
        excitation_wavelength = round(float(channel.find('ExcitationWavelength').text))
        filename = os.path.basename(path)
        channel_name = get_channel_from_filename_excitation_wavelength(filename, excitation_wavelength)
        # print(channel_name, ": ", excitation_wavelength)
        if channels is None or channel_name in channels:
            data[channel_name] = im.data[t_start:t_end, i, 0]

    return data


def get_channel_from_filename_excitation_wavelength(filename: str, excitation_wavelength: int):
    mito_aso = {"MtOr": 561, "MtDr": 642}
    drp1_aso = {"mEGFP": 488, "Halo?.DRP1": 642}
    mtdna_aso = {"TFAM": 561, "Sybr?.Gold": 488}

    for key in mito_aso.keys():
        regex = re.compile(f".*{key.lower()}.*")
        if (regex.match(filename.lower()) is not None) and excitation_wavelength == mito_aso[key]:
            return "mito"

    for key in drp1_aso.keys():
        regex = re.compile(f".*{key.lower()}.*")
        if (regex.match(filename.lower()) is not None) and excitation_wavelength == drp1_aso[key]:
            return "drp1"

    for key in mtdna_aso.keys():
        regex = re.compile(f".*{key.lower()}.*")
        if (regex.match(filename.lower()) is not None) and excitation_wavelength == mtdna_aso[key]:
            return "mtdna"


def main(argv):
    dir_name = ''
    opts, args = getopt.getopt(argv, "hd:", ["dir="])
    for opt, arg in opts:
        if opt == '-h':
            print("""extract.py -d <dir>
                  Extracts the trajectories of the drp1, the skeleton of the mitochondria
                  and put them together. Saves the trajectories as the same file name with
                  '_tracking.csv' as ending and the mitochondrial skeleton with the
                  '_mitoskel.csv' ending.
                  
                  Note: Currently only up to 4GB tif files are supported.
                  """)
            sys.exit()
        elif opt in ("-d", "--dir"):
            dir_name = arg
    extract(dir_name)


if __name__ == "__main__":
    main(sys.argv[1:])
