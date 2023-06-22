# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#
# A tool to create a procedural generated volume saved in
# MHD (https://itk.org/Wiki/ITK/MetaIO/Documentation) format.
#

import os
import argparse
import math
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a procedural generated volume and save as a MHD file.')

    parser.add_argument('output')

    args = parser.parse_args()
    
    size = 128
    
    # write the header
    with open(args.output + '.mhd', 'wt') as out_header:
        out_header.write(('ObjectType = Image\n'
            'NDims = 3\n'
            'BinaryData = True\n'
            'BinaryDataByteOrderMSB = False\n'
            'CompressedData = False\n'
            'TransformMatrix = 1 0 0 0 1 0 0 0 1\n'
            'Offset = 0 0 0\n'
            'CenterOfRotation = 0 0 0\n'
            'ElementSpacing = 1 1 1\n'
            'DimSize = ' + str(size) + ' ' + str(size) + ' ' + str(size) + '\n'
            'AnatomicalOrientation = ???\n'
            'ElementType = MET_UCHAR\n'
            'ElementDataFile = ' + args.output + '.raw\n'))

    density = np.zeros((size, size, size), dtype=np.uint8)
    for z in range(size):
        for y in range(size):
            for x in range(size):
                density[z][y][x] = int(
                    math.fabs(math.sin((float(x) / size) * 5.0 * math.pi)) *
                    math.fabs(math.sin((float(y) / size) * 4.0 * math.pi)) *
                    math.fabs(math.sin((float(z) / size) * 2.0 * math.pi)) * 255.0 + 0.5)

    with open(args.output + '.raw', 'w') as out_data:
        density.tofile(out_data)
