/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// This file contains the javascript that is run when the notebook is loaded.
// It contains some requirejs configuration and the `load_ipython_extension`
// which is required for any notebook extension.
//
// Some static assets may be required by the custom widget javascript. The base
// url for the notebook is not known at build time and is therefore computed
// dynamically.
__webpack_public_path__ =
    document.querySelector("body").getAttribute("data-base-url") +
    "nbextensions/clara-viz-widgets";

// Configure requirejs
if (window.require) {
    window.require.config({
        map: {
            "*": {
                "clara-viz-widgets": "nbextensions/clara-viz-widgets/index",
            },
        },
    });
}

// Export the required load_ipython_extension
module.exports = {
    load_ipython_extension: function () { },
};
