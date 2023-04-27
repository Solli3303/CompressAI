# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from compressai.registry import register_dataset
from aicsimageio.writers import OmeTiffWriter

from monai.transforms import SpatialPad
from aicsimageio import AICSImage
import numpy as np

@register_dataset("ImageFolder")
class ImageFolder(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, transform=None, split="train"):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = sorted(f for f in splitdir.iterdir() if f.is_file())

        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `torch.Tensor` or transformed `torch.Tensor`.
        """
        # Load the uint16 image and extraxt 3D array
        img = AICSImage(self.samples[index]).get_image_data("ZYX") 
        # Convert to float32 to fit model
        img = img.astype(np.float32)
        # Rescale unint16 values to [0,1]
        img = img / 65535
        # Expand dimension
        img = np.expand_dims(img, axis=0)
        # Apply optional transformations
        if self.transform:
            img= self.transform(img)
        # Apply optional padding to fit z-axis to model shape
        #if img.shape[1]<64:
        #    img = SpatialPad(spatial_size=(64,128,128))(img)
        # Clamp values to [0,1] before handing to model
        img = img.clamp_(0, 1)
        #OmeTiffWriter.save(img.squeeze(0).numpy(), f'/mnt/eternus/users/Jan/image_u_u_{index}.tiff', dim_order='ZYX')
        return img 

    def __len__(self):
        return len(self.samples)
