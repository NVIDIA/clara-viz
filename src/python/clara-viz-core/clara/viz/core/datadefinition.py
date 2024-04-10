# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

from .importutil import optional_import
from collections import defaultdict
from pathlib import Path
from packaging import version
import json

# Monai Train has a special ITK installation, set 'allow_namespace_pkg' for the import to work there
itk, has_itk = optional_import("itk", allow_namespace_pkg=True)
numpy, _ = optional_import("numpy")
cucim, has_cucim = optional_import("cucim")
DLDataTypeCode, _ = optional_import("cucim.clara._cucim", name="DLDataTypeCode")


class DataDefinition():
    """Defines the data used by the renderer.

    Attributes:
        arrays: A list of 'Array' elements holding the volume data
        settings: The render settings
        fetch_func: A function to be called on demand data fetches
    """

    def __init__(self, path: Path = None, dimension_order=""):
        """
        Construct a DataDefinition object

        Args:
            path: data file to load (optional)
            dimension_order: a string defining the data organization and format, if not provided it's set to 'DXYZ'
                for three dimensional volume data and to 'CXY' for two dimensional image data (optional)
        """
        self.arrays = []
        self.settings = {}
        self.fetch_func = None
        if path is not None:
            self.append(path, dimension_order)

    class Array():
        """Defines one array.

        Attributes:
            levels: array of numpy arrays with the data for each level for multi-dimensional data, an array
                    with a single numpy array for other (e.g. volume) data
            element_sizes: Physical size of an element for each level. The order is defined by the 'dimension_order' field. For
                           elements which have no physical size like 'M' or 'T' the corresponding value is 1.0.
                           For multi-dimensional data this is an array of element sizes, else an array with a single
                           element.
                           Default: [1.0, 1.0, ...]
            dimension_order: A string defining the data organization and format. Each character defines
                   a dimension starting with the fastest varying axis and ending with the
                   slowest varying axis. For example a 2D color image is defined as 'CXY',
                   a time sequence of density volumes is defined as 'DXYZT'.
                   Each character can occur only once. Either one of the data element
                   definition characters 'C', 'D' or 'M' and the 'X' axis definition has to
                   be present.
                   - X: width
                   - Y: height
                   - Z: depth
                   - T: time
                   - I: sequence
                   - C: RGB(A) color
                   - D: density
                   - M: mask
            permute_axes: Permutes the given data axes, e.g. to swap x and y of a 3-dimensional
                          density array specify [0, 2, 1, 3]
            flip_axes: Flips the given axes, e.g. to flip the x axis of a 3-dimensional
                       density array specify [False, True, False, False]
        """

        def __init__(self, array=None, dimension_order="", order=""):
            """
            Construct an array

            Args:
                array: numpy array with the data (optional)
                dimension_order: a string defining the data organization and format (optional)
                order: deprecated since 0.2, use the 'dimension_order' argument instead
            """
            self.levels = []
            if array is not None:
                self.levels.append(array)
            self.element_sizes = []
            if not dimension_order:
                self.dimension_order = order
            else:
                self.dimension_order = dimension_order
            self.permute_axes = []
            self.flip_axes = []

        @property
        def array(self):
            """
            .. deprecated:: 0.2
                use the 'levels' array instead

            Get the numpy array

            Returns:
                numpy array
            """
            return self.levels[0] if len(self.levels) >= 1 else None

        @array.setter
        def array(self, value):
            """
            .. deprecated:: 0.2
                use the 'levels' array instead

            Set the numpy array
            """
            if len(self.levels) == 0:
                self.levels.append(value)
            else:
                self.levels[0] = value

        @property
        def order(self):
            """
            .. deprecated:: 0.2
                use the 'dimension_order' attribute instead

            Get the dimension order

            Returns:
                A string defining the data organization and format
            """
            return self.dimension_order

        @order.setter
        def order(self, value):
            """
            .. deprecated:: 0.2
                use the 'dimension_order' attribute instead

            Set the dimension order
            """
            self.dimension_order = value

        @property
        def element_size(self):
            """
            .. deprecated:: 0.2
                use the 'element_sizes' array instead

            Get the physical size of each element

            Returns:
                physical size of each element
            """
            if len(self.element_sizes) == 0:
                self.element_sizes.append([])
            return self.element_sizes[0]

        @element_size.setter
        def element_size(self, value):
            """
            .. deprecated:: 0.2
                use the 'element_sizes' array instead

            Set physical size of each element
            """
            if len(self.element_sizes) == 0:
                self.element_sizes.append(value)
            else:
                self.element_sizes[0] = value

        def _parse_cucim(self, path: Path):
            """Try to read a file with cuCIM

            Args:
                path: Path to file.

            Returns:
                'True' if the file had been successfully parsed
            """

            # check if cuCim is installed
            if not has_cucim:
                return False

            # open the image
            try:
                self._img = cucim.CuImage(path)
            except:
                return False

            if not self._img.is_loaded:
                return False

            # cucim stores everything in reverse order compared to us, have to take care of this when using cucim values
            if self.dimension_order and self._img.dims[::-1] != self.dimension_order:
                raise Exception(
                    f'{path}: Unexpected data organization, expected {self.dimension_order} dimension order but file has dimension order {self._img.dims[::-1]}')

            # TODO derive from self._img.direction and self._img.coord_sys
            for dim in range(self._img.ndim):
                self.permute_axes.append(dim)
                self.flip_axes.append(False)

            numpy_dtype = None
            if self._img.dtype.code == DLDataTypeCode.DLInt:
                if self._img.dtype.bits == 8:
                    numpy_dtype = numpy.int8
                elif self._img.dtype.bits == 16:
                    numpy_dtype = numpy.int16
                elif self._img.dtype.bits == 32:
                    numpy_dtype = numpy.int32
            elif self._img.dtype.code == DLDataTypeCode.DLUInt:
                if self._img.dtype.bits == 8:
                    numpy_dtype = numpy.uint8
                elif self._img.dtype.bits == 16:
                    numpy_dtype = numpy.uint16
                elif self._img.dtype.bits == 32:
                    numpy_dtype = numpy.uint32
            elif self._img.dtype.code == DLDataTypeCode.DLFloat:
                if self._img.dtype.bits == 32:
                    numpy_dtype = numpy.float32

            if numpy_dtype is None:
                raise Exception(f"Unhandled data type {self._img.dtype}")

            self.levels = []
            self.element_sizes = []
            for level in range(self._img.resolutions["level_count"]):
                spacing = self._img.spacing()
                downsamples = self._img.resolutions["level_downsamples"][level]

                if (version.parse(cucim.__version__) > version.parse("22.07")):
                    # before cucim 22.08 spacing was fixed to 1.0, see https://github.com/rapidsai/cucim/issues/333
                    # this had been fixed with 22.08, now it's in micrometer and we have millimeter
                    if (self._img.spacing_units()[0] == "micrometer"):
                        downsamples /= 1000
                    else:
                        print(f"Unhandled spacing units {self._img.spacing_units[0]}, using millimeter")

                # scale spacing (exclude color dimension)
                for index in range(self._img.ndim - 1):
                    spacing[index] *= downsamples
                # spacing is in cucim 'dims' order 'ZXC', element_sizes is in clara-viz 'dimension_order' order 'CXZ', need to flip
                self.element_sizes.append(spacing[::-1])

                # build resolution ('level_dimensions' is in format x,y), no need to flip
                resolution = self._img.resolutions["level_dimensions"][level][::-1]
                resolution = resolution + (self._img.shape[-1],)

                # create an empty array with the resolution and data type
                self.levels.append(numpy.ndarray(resolution, numpy_dtype))

            return True

        def _fetch_cucim(self, context, level_index, offset, size, fetch_callback_func):
            """ This function is called to trigger on demand data fetches

            Args:
                context: internal context of the fetch call, pass to fetch callback function
                level_index: index of the level to fetch data for
                offset: offset into the level to fetch data for
                size: size of the data to fetch
                fetch_callback_func: callback function, called when data is received
            """
            if self._img is None:
                raise Exception('The image to read from is not defined')

            # in cucim the offset is level 0 based, the offset we get is based on the current level, need to convert
            location = [
                int(offset[1] * self._img.resolutions["level_dimensions"][0][0] /
                    self._img.resolutions["level_dimensions"][level_index][0]),
                int(offset[2] * self._img.resolutions["level_dimensions"][0][1] /
                    self._img.resolutions["level_dimensions"][level_index][1])
            ]

            region = self._img.read_region(location, size=size[1:3:], level=level_index)
            if not region.__array_interface__['strides'] is None:
                print('Strided cuCIM data is not yet supported')
                return False

            fetch_callback_func(context, level_index, offset, size, numpy.asarray(region))
            return True

        def _parse_itk(self, path: Path):
            """Try to read a file with ITK

            Args:
                path: Path to file.

            Returns:
                'True' if the file had been successfully parsed
            """

            # check if ITK is installed
            if not has_itk:
                return False

            # convert ImageIOBase type to pixel type
            ComponentTypeResolver = defaultdict(lambda: itk.F, {
                itk.CommonEnums.IOComponent_FLOAT: itk.F,
                itk.CommonEnums.IOComponent_LONG: itk.SL,
                itk.CommonEnums.IOComponent_ULONG: itk.UL,
                itk.CommonEnums.IOComponent_SHORT: itk.SS,
                itk.CommonEnums.IOComponent_USHORT: itk.US,
                itk.CommonEnums.IOComponent_CHAR: itk.SC,
                itk.CommonEnums.IOComponent_UCHAR: itk.UC
            })

            # Use image io to get information on the volume
            io = itk.ImageIOFactory.CreateImageIO(path, itk.CommonEnums.IOFileMode_ReadMode)
            if io is None:
                raise IOError(f'Failed to load file {path}')
            io.SetFileName(path)
            io.ReadImageInformation()

            dimensions = io.GetNumberOfDimensions()
            componentType = io.GetComponentType()

            self.element_size = [1.0]
            for dim in range(dimensions):
                self.element_size.append(io.GetSpacing(dim))

            pixelType = ComponentTypeResolver[componentType]
            imageType = itk.Image[pixelType, dimensions]

            if dimensions == 3:
                # get permute and flip axes values for volumes
                orient_filter = itk.OrientImageFilter[imageType, imageType].New()

                dir = [io.GetDirection(0), io.GetDirection(1), io.GetDirection(2)]
                np_dir_vnl = itk.vnl_matrix_from_array(numpy.array(dir))
                direction = itk.Matrix[itk.D, 3, 3](np_dir_vnl)
                orient_filter.SetGivenCoordinateDirection(direction)
                # ITK 5.3.0 changed the spatial orientation binding enums
                if (version.parse(itk.__version__) >= version.parse("5.3.0")):
                    orient_filter.SetDesiredCoordinateOrientation(
                        itk.SpatialOrientationEnums().ValidCoordinateOrientations_ITK_COORDINATE_ORIENTATION_RIP)
                else:
                    orient_filter.SetDesiredCoordinateOrientation(
                        itk.itkSpatialOrientationPython.itkSpatialOrientation_ITK_COORDINATE_ORIENTATION_RIP)

                self.permute_axes.append(0)
                self.flip_axes.append(False)
                for dim in range(io.GetNumberOfDimensions()):
                    self.permute_axes.append(orient_filter.GetPermuteOrder()[dim] + 1)
                    self.flip_axes.append(orient_filter.GetFlipAxes()[dim])
            else:
                for index in range(dimensions):
                    self.permute_axes.append(index)
                    self.flip_axes.append(False)

            # create the reader
            __itk_reader = itk.ImageFileReader[imageType].New()
            __itk_reader.SetFileName(path)
            __itk_reader.Update()

            self.levels = [itk.GetArrayViewFromImage(__itk_reader.GetOutput())]

            if io.GetNumberOfComponents() == 1:
                # for single component data the numpy array does not have an axis, add it
                self.levels[0] = self.levels[0][..., numpy.newaxis]

            return True

    def __fetch_cucim(self, context, array_id, level_index, offset, size, fetch_callback_func):
        """ This function is called to trigger on demand data fetches

        Args:
            context: internal context of the fetch call, pass to fetch callback function
            array_id: id of the array to fetch data from
            level_index: index of the level to fetch data for
            offset: offset into the level to fetch data for
            size: size of the data to fetch
            fetch_callback_func: callback function, called when data is received
        """
        for array in self.arrays:
            if array.dimension_order == array_id:
                return array._fetch_cucim(context, level_index, offset, size, fetch_callback_func)

        raise Exception(f'Array with id {array_id} not found')

    def append(self, path: Path, dimension_order=""):
        """
        Append a file to the DataDefinition.

        Args:
            path: path to file
            dimension_order: a string defining the data organization and format, if not provided it's set to 'DXYZ'
                for three dimensional volume data and to 'CXY' for two dimensional image data
        """
        array = self.Array()
        array.dimension_order = dimension_order

        # try to read 2D images (e.g. 'CXY') with cucim because that's the only format cucim supports
        if (not dimension_order or len(dimension_order) == 3) and array._parse_cucim(path):
            self.arrays.append(array)
            # default to 'CXY' if dimension order is not specified
            if not array.dimension_order:
                array.dimension_order = 'CXY'
            self.fetch_func = self.__fetch_cucim
            return

        # then try to read everything else with ITK
        if array._parse_itk(path):
            self.arrays.append(array)
            # deduce dimension order if not specified
            if not array.dimension_order:
                if len(array.levels[0].shape) == 4:
                    array.dimension_order = 'DXYZ'
                elif len(array.levels[0].shape) == 3:
                    array.dimension_order = 'CXY'
                else:
                    raise Exception(f'Could not deduce dimension order from file, please specify the order')
            return

        raise Exception(f'The format of the file {path} is not supported')

    def load_settings(self, path: Path):
        """
        Read settings from a JSON file

        Args:
            path: path to the JSON file to read
        """
        with open(path) as f:
            self.settings = json.load(f)
