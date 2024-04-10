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

import ipywidgets as widgets
import json
import copy
from traitlets import Float, Int, Unicode

from ._version import __version__

import clara.viz.core

# See js/lib/widget.js for the frontend counterpart to this file.


@widgets.register
class Widget(widgets.DOMWidget):
    """An interactive render widget.

    Attributes:
        settings: a dict with render settings
        width: width of the video stream
        height: height of the video stream
        frame_rate: frame rate of the video stream
        bit_rate: bit rate of the video stream
    """

    # Name of the widget view class in front-end
    _view_name = Unicode('WidgetView').tag(sync=True)

    # Name of the widget model class in front-end
    _model_name = Unicode('WidgetModel').tag(sync=True)

    # Name of the front-end module containing widget view
    _view_module = Unicode('clara-viz-widgets').tag(sync=True)

    # Name of the front-end module containing widget model
    _model_module = Unicode('clara-viz-widgets').tag(sync=True)

    # Version of the front-end module containing widget view
    _view_module_version = Unicode(f'^{__version__}').tag(sync=True)
    # Version of the front-end module containing widget model
    _model_module_version = Unicode(f'^{__version__}').tag(sync=True)

    # Widget specific properties.
    # video
    width = Int(640, help='Width of the video stream').tag(sync=False)
    height = Int(480, help='Height of the video stream').tag(sync=False)
    frame_rate = Float(30.0, help='Frame rate of the video stream').tag(sync=False)
    bit_rate = Int(8 * 1024 * 1024, help='Bit rate of the video stream').tag(sync=False)

    # internal

    # render settings (synced with widget)
    _settings = Unicode('{}').tag(sync=True)
    # dataset info (synced with widget)
    _dataset_info = Unicode('{}').tag(sync=True)

    def __init__(self, renderer=None, data_definition=None, **kwargs):
        """Construct a widget

        Args:
            renderer: renderer to use, if not specified the widget is creating a renderer (optional)
            data_definition: dataset definition (optional)
        """

        # call super class init
        super(Widget, self).__init__(**kwargs)

        # video stream
        self._video_stream = False
        # video should play
        self._video_play = False
        # video is visible
        self._video_visible = False

        if renderer:
            # use the provided renderer
            self._renderer = renderer
        else:
            # create the renderer
            self._renderer = clara.viz.core.Renderer()

        if data_definition:
            self.select_data_definition(data_definition)
        else:
            # get the current settings from the renderer
            self.settings = self._renderer.get_settings()
            # update the settings synced with the widget
            self._settings = json.dumps(self.settings)

        # get the current array configuration from the renderer
        size = []
        element_size = []
        permute_axes = []
        arrays = self._renderer.get_arrays()
        if (len(arrays)):
            size = list(arrays[0].levels[0].shape)
            size.reverse()
            # for single component data the 'shape' does not have an axis, add it
            if len(size) < len(arrays[0].dimension_order):
                size.insert(0, 1)
            element_size = arrays[0].element_sizes[0]
            permute_axes = arrays[0].permute_axes

        # update the dataset info synced with the widget
        self.__update_dataset_info(size, element_size, permute_axes)

        # start video stream when the widget is displayed
        self.on_displayed(lambda widget, **kwargs: self.__on_displayed())

        # custom message handling
        self.on_msg(self.__on_msg)

        # observers
        self.observe(lambda change: self.__video_config(), ['width', 'height', 'frame_rate', 'bit_rate'])

    def select_data_definition(self, data_definition: clara.viz.core.DataDefinition):
        """Select data from a DataDefinition.

        Args:
            dataset_definition: a DataDefinition object
        """

        if len(data_definition.arrays) == 0:
            raise Exception('No arrays are defined.')

        # reset all settings
        self._renderer.reset()

        # set the arrays
        arrays = []
        for array in data_definition.arrays:
            arrays.append(clara.viz.core.Array(levels=array.levels, dimension_order=array.dimension_order, permute_axes=array.permute_axes,
                          flip_axes=array.flip_axes, element_sizes=array.element_sizes))
        self._renderer.set_arrays(arrays, data_definition.fetch_func)

        first_array = data_definition.arrays[0]
        size = list(first_array.levels[0].shape)
        size.reverse()
        # for single component data the 'shape' does not have an axis, add it
        if len(size) < len(first_array.dimension_order):
            size.insert(0, 1)
        self.__update_dataset_info(size, first_array.element_sizes[0], first_array.permute_axes)

        if data_definition.settings:
            self.settings = copy.deepcopy(data_definition.settings)
            self.set_settings()
        else:
            # let the renderer deduce the settings for the defined dataset
            self._renderer.deduce_settings()

        # the renderer might fix the settings, get them back
        self.settings = self._renderer.get_settings()
        # update the settings synced with the widget
        self._settings = json.dumps(self.settings)

    def set_settings(self):
        """Update the render settings."""

        # update the settings synced with the widget
        self._settings = json.dumps(self.settings)

        # pause the video stream while updating settings
        self._video_play = False
        self.__on_video_state_change()

        self._renderer.set_settings(self.settings)

        self._video_play = True
        self.__on_video_state_change()

    def __update_dataset_info(self, size, element_size, permute_axes):
        # update dataset info
        dataset_info = {
            'size': {'x': 1, 'y': 1, 'z': 1},
            'elementSize': {'x': 1.0, 'y': 1.0, 'z': 1.0},
        }
        if (len(size) > 0):
            permuted_size = [size[index] for index in permute_axes]
            if (len(permuted_size) > 1):
                dataset_info['size']['x'] = permuted_size[1]
            if (len(size) > 2):
                dataset_info['size']['y'] = permuted_size[2]
            if (len(size) > 3):
                dataset_info['size']['z'] = permuted_size[3]

        if (len(element_size) > 0):
            permuted_element_size = [element_size[index] for index in permute_axes]
            if (len(element_size) > 1):
                dataset_info['elementSize']['x'] = permuted_element_size[1]
            if (len(element_size) > 2):
                dataset_info['elementSize']['y'] = permuted_element_size[2]
            if (len(element_size) > 3):
                dataset_info['elementSize']['z'] = permuted_element_size[3]

        self._dataset_info = json.dumps(dataset_info)

    def __on_displayed(self):
        # create the video stream
        self._video_stream = self._renderer.create_video_stream(self.__video_stream_callback)

        # configure the video
        self.__video_config()

        # start the video stream
        self._video_play = True
        self.__on_video_state_change()

    def __video_config(self):
        if self._video_stream:
            self._video_stream.configure(width=self.width, height=self.height,
                                         frame_rate=self.frame_rate, bit_rate=self.bit_rate)

    def __on_video_state_change(self):
        if self._video_stream:
            if self._video_play and self._video_visible:
                self._video_stream.play()
            else:
                self._video_stream.pause()

    def __on_msg(self, widget, content, buffers):
        if content['msg_type'] == 'camera_update':
            self.settings['Cameras'] = copy.deepcopy(content['contents'])
            self._renderer.merge_settings(self.settings)
        elif content['msg_type'] == 'data_view_update':
            self.settings['DataViews'] = copy.deepcopy(content['contents'])
            self._renderer.merge_settings(self.settings)
        elif content['msg_type'] == 'video_visible':
            self._video_visible = content['contents']
            self.__on_video_state_change()
        else:
            print('Received unhandled message ', content)

    def __video_stream_callback(self, data, new_stream):
        # create a copy of the data and send to widget
        data_copy = bytes(data)
        self.send('stream', [data_copy])
