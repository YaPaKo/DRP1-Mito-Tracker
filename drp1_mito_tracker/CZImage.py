import os
import re
from typing import Union

import cupy as cp
import numpy as np
from aicsimageio import AICSImage


class CZImage:
    MITO = "mito"
    DRP1 = "drp1"
    MTDNA = "mtdna"
    C = {MITO, DRP1, MTDNA}

    def __init__(self, path: str, channel: str = None):
        self.path = path
        self.filename = os.path.basename(path)
        self.data = AICSImage(path)
        self.channel = channel
        self.channels = {}

        for i, data_channel in enumerate(
                self.data.metadata.find('Metadata')
                                  .find('Information')
                                  .find('Image')
                                  .find('Dimensions')
                                  .find('Channels')
        ):
            excitation_wavelength = round(float(data_channel.find('ExcitationWavelength').text))
            channel_name = self.get_channel_from_filename_excitation_wavelength(excitation_wavelength)
            self.channels[channel_name] = i

    def __len__(self):
        return self.data.dims.T

    def set_channel(self, channel):
        self.channel = channel

    def np(self, t: int, channel=None) -> np.array:
        if channel is None:
            channel = self.channel
        if isinstance(channel, int):
            return self.data.get_image_dask_data("YX", Z=0, T=t, C=channel).compute()
        elif isinstance(channel, str):
            if channel not in self.C:
                raise KeyError("Invalid channel. Please provide a valid channel.")
            return self.data.get_image_dask_data("YX", Z=0, T=t, C=self.channels[channel]).compute()
        else:
            raise ValueError("Value should be of type str or int.")

    def cp(self, t: int, channel=None) -> cp.array:
        return cp.array(self.np(t, channel))

    def get_channel_from_filename_excitation_wavelength(self, excitation_wavelength: int):
        mito_aso = {"MtOr": 561, "MtDr": 642}
        drp1_aso = {"mEGFP": 488, "Halo?.DRP1": 642}
        mtdna_aso = {"TFAM": 561, "Sybr?.Gold": 488}

        for key in mito_aso.keys():
            regex = re.compile(f".*{key.lower()}.*")
            if (regex.match(self.filename.lower()) is not None) and excitation_wavelength == mito_aso[key]:
                return "mito"

        for key in drp1_aso.keys():
            regex = re.compile(f".*{key.lower()}.*")
            if (regex.match(self.filename.lower()) is not None) and excitation_wavelength == drp1_aso[key]:
                return "drp1"

        for key in mtdna_aso.keys():
            regex = re.compile(f".*{key.lower()}.*")
            if (regex.match(self.filename.lower()) is not None) and excitation_wavelength == mtdna_aso[key]:
                return "mtdna"
