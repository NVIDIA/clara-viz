/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION. All rights reserved.
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

import Stream from "./Stream.js";

const grpc = {};
grpc.web = require("grpc-web");

const pako = require('pako');

const {
    RenderServerClient,
} = require("./nvidia/claraviz/cinematic/v1/render_server_grpc_web_pb.js");
const { VideoClient } = require("./nvidia/claraviz/video/v1/video_grpc_web_pb.js");

const proto = {};
proto.camera = require("./nvidia/claraviz/core/camera_pb.js");
proto.types = require("./nvidia/claraviz/core/types_pb.js");
proto.video = require("./nvidia/claraviz/video/v1/video_pb.js");
proto.renderServer = require("./nvidia/claraviz/cinematic/v1/render_server_pb.js");

const renderServerHost =
    window.location.protocol + "//" + window.location.hostname + ":8082";

var rsClient = new RenderServerClient(renderServerHost);
var videoClient = new VideoClient(renderServerHost);

// video streams
var videoStreams = [];

// view angle in degree
var viewAngle = 0.0;
// volume physical size in meters
var physicalSize = [1.0, 1.0, 1.0];
// Camera distance
var cameraDistance = 1.0;
// Transfer function range
var transferFunctionMin = 0.0;
var transferFunctionMax = 100.0;
// is animation enabled?
var animated = false;
// is multi view enabled?
var multiView = false;

// steps per rotation
const STEPS = 144;
// field of view for cinematic view
const FOV = 35.0;
// field of view for slice view
const FOV_SLICE = 20.0;
// frames per second
const FPS = 30.0;

// timout id for letting the progress vanish
var endProgressTimeout = null;

/**
 * Send a gRPC request to the RenderServer and wait for the result.
 *
 * @param {*} request to send
 * @param {string} key of the function to call
 */
async function grpcClosureRSSync(request, key) {
    return await rsClient[key](request, {}, (err, response) => {
        if (err) {
            log("gRPC request ", key, " failed");
            log(err.code);
            log(err.message);
            log(response);
        }
    });
};

/**
 * Send a async gRPC request to the RenderServer.
 *
 * @param {*} request to send
 * @param {string} key of the function to call
 * @param {function} callback function to call when the request is done
 */
function grpcClosureRSAsync(request, key, callback = null) {
    rsClient[key](request, {}, (err, response) => {
        if (err) {
            log("gRPC request ", key, " failed");
            log(err.code);
            log(err.message);
            log(response);
        } else if (callback) {
            callback(response);
        }
    });
};

/**
 * Update the RenderServer camera.
 */
function updateCamera() {
    var cameraRequest = new proto.camera.CameraRequest();

    cameraRequest.setName("CinematicCamera");
    var eye = new proto.types.Float3();
    eye.setX(Math.sin(viewAngle / 360.0 * (2.0 * Math.PI)) * cameraDistance);
    eye.setY(0.0);
    eye.setZ(-Math.cos(viewAngle / 360.0 * (2.0 * Math.PI)) * cameraDistance);
    cameraRequest.setEye(eye);
    var lookAt = new proto.types.Float3();
    lookAt.setX(0.0);
    lookAt.setY(0.0);
    lookAt.setZ(0.0);
    cameraRequest.setLookAt(lookAt);
    var up = new proto.types.Float3();
    up.setX(0.0);
    up.setY(1.0);
    up.setZ(0.0);
    cameraRequest.setUp(up);
    cameraRequest.setFieldOfView(FOV);
    cameraRequest.setPixelAspectRatio(1.0);

    grpcClosureRSSync(cameraRequest, "camera");

    var cameraRequest = new proto.camera.CameraRequest();

    cameraRequest.setName("SliceFrontCamera");
    var eye = new proto.types.Float3();
    eye.setX(0.0);
    eye.setY(0.0);
    eye.setZ((viewAngle / 360.0 + 0.5) * physicalSize[2]);
    cameraRequest.setEye(eye);
    var lookAt = new proto.types.Float3();
    lookAt.setX(0.0);
    lookAt.setY(0.0);
    lookAt.setZ((viewAngle / 360.0 - 0.5) * physicalSize[2]);
    cameraRequest.setLookAt(lookAt);
    var up = new proto.types.Float3();
    up.setX(0.0);
    up.setY(1.0);
    up.setZ(0.0);
    cameraRequest.setUp(up);
    cameraRequest.setFieldOfView(FOV_SLICE);
    cameraRequest.setPixelAspectRatio(1.0);

    grpcClosureRSSync(cameraRequest, "camera");

    var cameraRequest = new proto.camera.CameraRequest();

    cameraRequest.setName("SliceRightCamera");
    var eye = new proto.types.Float3();
    eye.setX((viewAngle / 360.0 + 0.5) * physicalSize[0]);
    eye.setY(0.0);
    eye.setZ(0.0);
    cameraRequest.setEye(eye);
    var lookAt = new proto.types.Float3();
    lookAt.setX((viewAngle / 360.0 - 0.5) * physicalSize[0]);
    lookAt.setY(0.0);
    lookAt.setZ(0.0);
    cameraRequest.setLookAt(lookAt);
    var up = new proto.types.Float3();
    up.setX(0.0);
    up.setY(1.0);
    up.setZ(0.0);
    cameraRequest.setUp(up);
    cameraRequest.setFieldOfView(FOV_SLICE);
    cameraRequest.setPixelAspectRatio(1.0);

    grpcClosureRSSync(cameraRequest, "camera");

    var cameraRequest = new proto.camera.CameraRequest();

    cameraRequest.setName("SliceTopCamera");
    var eye = new proto.types.Float3();
    eye.setX(0.0);
    eye.setY((viewAngle / 360.0 + 0.5) * physicalSize[1]);
    eye.setZ(0.0);
    cameraRequest.setEye(eye);
    var lookAt = new proto.types.Float3();
    lookAt.setX(0.0);
    lookAt.setY((viewAngle / 360.0 - 0.5) * physicalSize[1]);
    lookAt.setZ(0.0);
    cameraRequest.setLookAt(lookAt);
    var up = new proto.types.Float3();
    up.setX(0.0);
    up.setY(0.0);
    up.setZ(1.0);
    cameraRequest.setUp(up);
    cameraRequest.setFieldOfView(FOV_SLICE);
    cameraRequest.setPixelAspectRatio(1.0);

    grpcClosureRSSync(cameraRequest, "camera");
}

/**
 * Update the RenderServer view.
 */
function updateView() {
    var viewRequest = new proto.renderServer.ViewRequest();
    viewRequest.setName("CinematicView");
    viewRequest.setMode(proto.renderServer.ViewRequest.ViewMode.CINEMATIC);
    viewRequest.setStreamName("CinematicStream");
    viewRequest.setCameraName("CinematicCamera");
    grpcClosureRSAsync(viewRequest, "view");

    var viewRequest = new proto.renderServer.ViewRequest();
    viewRequest.setName("SliceFrontView");
    viewRequest.setMode(proto.renderServer.ViewRequest.ViewMode.SLICE);
    if (multiView) {
        viewRequest.setStreamName("SliceFrontStream");
    }
    viewRequest.setCameraName("SliceFrontCamera");
    grpcClosureRSAsync(viewRequest, "view");

    var viewRequest = new proto.renderServer.ViewRequest();
    viewRequest.setName("SliceRightView");
    viewRequest.setMode(proto.renderServer.ViewRequest.ViewMode.SLICE);
    if (multiView) {
        viewRequest.setStreamName("SliceRightStream");
    }
    viewRequest.setCameraName("SliceRightCamera");
    grpcClosureRSAsync(viewRequest, "view");

    var viewRequest = new proto.renderServer.ViewRequest();
    viewRequest.setName("SliceTopView");
    viewRequest.setMode(proto.renderServer.ViewRequest.ViewMode.SLICE);
    if (multiView) {
        viewRequest.setStreamName("SliceTopStream");
    }
    viewRequest.setCameraName("SliceTopCamera");
    grpcClosureRSAsync(viewRequest, "view");
}

/**
 * Called when animation mode is enabled
 */
function runAnimation() {
    viewAngle = (viewAngle + 360.0 / STEPS) % 360.0;
    document.getElementById("angle").value = viewAngle;
    updateCamera();
}

/**
 * Add an entry to the log.
 *
 * @param {string} str
 */
function log(str) {
    const log = document.getElementById("log");
    log.value += str + "\n";
    if (log.selectionStart == log.selectionEnd) {
        log.scrollTop = log.scrollHeight;
    }
}

/**
 * Update the progress bar
 *
 * @param {number} percent
 */
function updateProgress(percent) {
    const progress = document.querySelector('.percent');
    progress.style.width = percent + '%';
    progress.textContent = percent + '%';
}

/**
 * Start progress.
 *
 * @param {*} msg message to add to log
 */
function startProgress(msg) {
    // clear any previous end progress timeout
    if (endProgressTimeout) {
        clearTimeout(endProgressTimeout);
    }

    log(msg);
    document.getElementById('progress_bar').className = 'loading';
    updateProgress(0);
    // switch to wait cursor
    document.getElementById("root").style.cursor = "wait";
}

/**
 * End progress.
 */
function endProgress() {
    // switch back to normal cursor
    document.getElementById("root").style.cursor = "pointer";
    // Ensure that the progress bar displays 100% at the end.
    updateProgress(100);
    // let it fade out after 2 seconds
    endProgressTimeout = setTimeout("document.getElementById('progress_bar').className='';", 2000);
}

/**
 * File load error handler.
 *
 * @param {*} event
 */
function onFileReaderError(event) {
    switch (event.target.error.code) {
        case event.target.error.NOT_FOUND_ERR:
            alert('File Not Found!');
            break;
        case event.target.error.NOT_READABLE_ERR:
            alert('File is not readable');
            break;
        case event.target.error.ABORT_ERR:
            break; // noop
        default:
            alert('An error occurred reading this file.');
    };
}

/**
 * File load progress handler.
 *
 * @param {*} event
 */
function onFileProgress(event) {
    // event is an ProgressEvent.
    if (event.lengthComputable) {
        var percentLoaded = Math.round((event.loaded / event.total) * 100);
        // Increase the progress bar length.
        if (percentLoaded < 100) {
            updateProgress(percentLoaded);
        }
    }
}

/**
 * Send a data slice to RenderServer.
 *
 * @param {array} rawData all the data
 * @param {number} slice slice index
 * @param {string} id data array id
 * @param {array} size array size
 * @param {number} elementSize size of one element in bytes
 */
function sendSlice(rawData, slice, id, size, elementSize) {
    var dataRequest = new proto.renderServer.DataRequest();

    dataRequest.setArrayId(id);
    var partialSize = Array.from(size);
    partialSize[3] = 1;
    dataRequest.setSizeList(partialSize);
    const offset = [0, 0, 0, slice];
    dataRequest.setOffsetList(offset);
    const sliceSize = partialSize[0] * partialSize[1] * partialSize[2] * partialSize[3] * elementSize;
    dataRequest.setData(rawData.slice(slice * sliceSize, (slice + 1) * sliceSize));

    grpcClosureRSSync(dataRequest, "data");
    updateProgress(Math.round(((slice + 1) / size[3]) * 100));
}

/**
 * Update the RenderServer transfer function.
 */
function updateTransferFunction() {
    document.getElementById("tf_min").value = transferFunctionMin;
    document.getElementById("tf_max").value = transferFunctionMax;

    var transferFunctionRequest = new proto.renderServer.TransferFunctionRequest();

    transferFunctionRequest.setBlendingProfile(proto.renderServer.TransferFunctionRequest.BlendingProfile.MAXIMUM_OPACITY);
    transferFunctionRequest.setDensityScale(100.0);
    transferFunctionRequest.setGlobalOpacity(1.0);
    transferFunctionRequest.setGradientScale(10.0);
    transferFunctionRequest.setShadingProfile(proto.renderServer.TransferFunctionRequest.ShadingProfile.HYBRID);

    var component = new proto.renderServer.TransferFunctionRequest.Component;

    var range = new proto.types.Range();
    range.setMin(transferFunctionMin / 100.0);
    range.setMax(transferFunctionMax / 100.0);
    component.setRange(range);
    component.setOpacityProfile(proto.renderServer.TransferFunctionRequest.Component.OpacityProfile.SQUARE);
    component.setOpacityTransition(0.2);
    component.setOpacity(1.0);
    component.setRoughness(0.5);
    component.setEmissiveStrength(0.0);
    var diffuse = new proto.types.Float3();
    diffuse.setX(0.7);
    diffuse.setY(0.7);
    diffuse.setZ(1.0);
    component.setDiffuseStart(diffuse);
    component.setDiffuseEnd(diffuse);
    var specular = new proto.types.Float3();
    specular.setX(1.0);
    specular.setY(1.0);
    specular.setZ(1.0);
    component.setSpecularStart(specular);
    component.setSpecularEnd(specular);
    var emissive = new proto.types.Float3();
    emissive.setX(0.0);
    emissive.setY(0.0);
    emissive.setZ(0.0);
    component.setEmissiveStart(emissive);
    component.setEmissiveEnd(emissive);

    transferFunctionRequest.addComponents(component);

    grpcClosureRSAsync(transferFunctionRequest, "transferFunction");
}

/**
 * Define the transfer function.
 *
 * @param {string} id data array id
 */
function defineTransferFunction(id) {

    // get a histogram of the density data, find the maximum and define a transform function
    // around that maximum
    log("Getting data histogram");
    var dataHistorgramRequest = new proto.renderServer.DataHistogramRequest();
    dataHistorgramRequest.setArrayId(id);
    grpcClosureRSAsync(dataHistorgramRequest, "dataHistogram", (response) => {
        const histogram = response.getDataList();
        // skip the first few elements of the histogram, this is usual the air around a scan
        var maxIndex = 5;
        var max = histogram[5];
        for (var i = maxIndex; i < histogram.length; i++) {
            if (histogram[i] > max) {
                maxIndex = i;
                max = histogram[i];
            }
        }
        const maxPosNormalized = maxIndex / (histogram.length - 1.0);
        log("Data histogram max is at " + maxPosNormalized);

        // define the transfer function around the histogram maximum
        transferFunctionMin = Math.max(0.0, maxPosNormalized - 0.025) * 100.0;
        transferFunctionMax = Math.min(1.0, maxPosNormalized + 0.025) * 100.0;
        updateTransferFunction();
    });
}

/**
 * Process the data read from the RAW file.
 *
 * @param {array} rawData all the data
 * @param {string} id data array id
 * @param {array} size volume size
 * @param {number} elementSize size of one element in bytes
 * @param {boolean} compressed flag indicating if raw data is compressed
 * @returns
 */
function processRAWData(rawData, id, size, elementSize, compressed) {
    if (compressed) {
        log("Data is compressed, uncompressing data...");
        try {
            rawData = pako.inflate(rawData);
        } catch (err) {
            log("Uncompressing failed with error: " + err);
            return;
        }
        log("...done");
    }
    // send the data, slice by slice, since there is a message size limit of 4MB
    startProgress("Sending volume data...");
    for (var z = 0; z < size[3]; z++) {
        // use a timeout function with 0 to run the upload asynchronously
        setTimeout(sendSlice, 0, rawData, z, id, size, elementSize);
    }
    setTimeout(endProgress, 0);
    setTimeout(defineTransferFunction, 0, id);
}

/**
 * Parse the MHD header.
 *
 * @param {string} mhdHeader
 * @param {File} rawFile
 */
function parseMHDHeader(mhdHeader, rawFile) {
    var size = [1];
    var elementSize = [1.0];
    var elementType;
    var elementTypeSize;
    var compressed = false;

    const lines = mhdHeader.split("\n");
    for (var i = 0; i < lines.length; i++) {
        // remove duplicate spaces
        const line = '' + lines[i].replace(/ +(?= )/g, '');
        // get parameter and value
        var [parameter, value] = line.split("=");
        // remove spaces at front and end
        parameter = parameter.trim();
        value = value ? value.trim() : "";
        switch (parameter) {
            case "NDims":
                if (parseInt(value) != 3) {
                    alert("Expected a three dimensional input, instead NDims is " + value);
                    return;
                }
                break;
            case "CompressedData":
                if (value == "True") {
                    compressed = true;
                } else if (value == "False") {
                    compressed = false;
                } else {
                    alert("Unhandled value " + value + " for tag 'CompressedData'");
                    return;
                }
                break;
            case "DimSize":
                value.split(" ").forEach(function (dimSize) {
                    size.push(parseInt(dimSize));
                });
                break;
            case "ElementSpacing":
                value.split(" ").forEach(function (spacing) {
                    elementSize.push(parseFloat(spacing));
                });
                break;
            case "ElementType":
                switch (value) {
                    case "MET_CHAR":
                        elementType = proto.renderServer.DataConfigRequest.Array.ElementType.INT8;
                        elementTypeSize = 1;
                        break;
                    case "MET_UCHAR":
                        elementType = proto.renderServer.DataConfigRequest.Array.ElementType.UINT8;
                        elementTypeSize = 1;
                        break;
                    case "MET_SHORT":
                        elementType = proto.renderServer.DataConfigRequest.Array.ElementType.INT16;
                        elementTypeSize = 2;
                        break;
                    case "MET_USHORT":
                        elementType = proto.renderServer.DataConfigRequest.Array.ElementType.UINT16;
                        elementTypeSize = 2;
                        break;
                    case "MET_INT":
                        elementType = proto.renderServer.DataConfigRequest.Array.ElementType.INT32;
                        elementTypeSize = 4;
                        break;
                    case "MET_UINT":
                        elementType = proto.renderServer.DataConfigRequest.Array.ElementType.UINT32;
                        elementTypeSize = 4;
                        break;
                    case "MET_FLOAT":
                        elementType = proto.renderServer.DataConfigRequest.Array.ElementType.FLOAT;
                        elementTypeSize = 4;
                        break;
                    default:
                        alert("Unexpected value for " + parameter + ": " + value);
                        return;
                }
                break;
            case "ElementDataFile":
                if (value != rawFile.name) {
                    alert("The file name of the .raw file does not match the file name of the data file given in the .mhd file");
                    return;
                }
                break;
        }
    };

    // configure the data
    var dataConfigRequest = new proto.renderServer.DataConfigRequest();
    var array = new proto.renderServer.DataConfigRequest.Array;
    var id = new proto.types.Identifier;
    id.setValue("density");
    array.setId(id);
    array.setDimensionOrder("DXYZ");
    array.setElementType(elementType);
    var level = new proto.renderServer.DataConfigRequest.Array.Level;
    level.setSizeList(size);
    level.setElementSizeList(elementSize);
    array.addLevels(level);

    dataConfigRequest.addArrays(array);

    grpcClosureRSSync(dataConfigRequest, "dataConfig");

    // calculate the volume physical size, element spacing is in mm, physical size is in meters
    for (var index = 0; index < 3; index++)
        physicalSize[index] = size[index + 1] * elementSize[index + 1] / 1000.0;
    const maxPhysicalSize = Math.max(...physicalSize);

    // calculate the camera distance for a given volume physical size and a field of view (multiply by 10% to better see the object)
    cameraDistance = (0.5 / Math.tan(FOV * 0.5 * Math.PI / 180.0)) * maxPhysicalSize * 1.1;
    updateCamera();

    // place the light
    var lightRequest = new proto.renderServer.LightRequest();

    lightRequest.setIndex(0);
    var position = new proto.types.Float3();
    position.setX(-2.0 * physicalSize[0]);
    position.setY(2.0 * physicalSize[1]);
    position.setZ(-2.0 * physicalSize[2]);
    lightRequest.setPosition(position);
    var direction = new proto.types.Float3();
    direction.setX(Math.sqrt(1.0 / 3.0));
    direction.setY(Math.sqrt(1.0 / 3.0));
    direction.setZ(Math.sqrt(1.0 / 3.0));
    lightRequest.setDirection(direction);
    lightRequest.setSize(maxPhysicalSize / 2.0);
    const lightDistanceSquared = (physicalSize[0] * physicalSize[0]) + (physicalSize[1] * physicalSize[1]) + (physicalSize[2] * physicalSize[2]);
    lightRequest.setIntensity(lightDistanceSquared * 10.0);
    var light = new proto.types.Float3();
    light.setX(1.0);
    light.setY(1.0);
    light.setZ(1.0);
    lightRequest.setColor(light);
    lightRequest.setEnable(proto.types.Switch.SWITCH_ENABLE);

    grpcClosureRSAsync(lightRequest, "light");

    // read the RAW data file
    var reader = new FileReader();
    reader.onerror = onFileReaderError;
    reader.onloadstart = function () { startProgress("Reading RAW volume data file") };
    reader.onprogress = onFileProgress;
    reader.onload = function () {
        endProgress();
        processRAWData(reader.result, id, size, elementTypeSize, compressed);
    }
    reader.readAsArrayBuffer(rawFile);
}

/**
 * Load a volume handler.
 *
 * @param {*} event
 */
window.onLoadVolumeFile = function (event) {
    if (!event.target.files[0]) {
        return;
    }

    var mhdFile;
    var rawFile;

    for (var i = 0; i < event.target.files.length; i++) {
        const file = event.target.files[i];
        switch (file.name.split('.').pop()) {
            case "mhd":
                mhdFile = file;
                break;
            case "raw":
                rawFile = file;
                break;
            default:
                alert("Unhandled file type " + file.name + ", has to be .mhd or .raw");
                return;
        }
    }

    if (!mhdFile || !rawFile) {
        alert("Both the .mhd file and the .raw file needs to be specified");
        return;
    }

    // read the MHD header file
    var reader = new FileReader();
    reader.onerror = onFileReaderError;
    reader.onloadstart = function () { startProgress("Reading MHD header file") };
    reader.onprogress = onFileProgress;
    reader.onload = function () {
        endProgress();
        parseMHDHeader(reader.result, rawFile);
    }
    reader.readAsText(mhdFile);
}

/**
 * Set the video stream size.
 */
function setVideoSize() {
    // configure the video(s) with the size of the video_area div element
    const videoDivElement = document.getElementById("video_area");
    for (var i = 0; i < videoStreams.length; i++) {
        videoStreams[i].ClientSize(
            videoDivElement.clientWidth / (multiView ? 2 : 1),
            videoDivElement.clientHeight / (multiView ? 2 : 1)
        );
    }
}

// camera animation interval
var cameraInterval;

/**
 * Toggle animation handler.
 */
window.onToggleAnimation = function () {
    if (animated) {
        // stop the animation
        clearInterval(cameraInterval);
        animated = false;
    } else {
        // start the animation (interval time is ms, call once per frame)
        cameraInterval = setInterval(runAnimation, 1000 / FPS);
        animated = true;
    }
};

/**
 * Toggle multi view handler.
 */
window.onToggleMultiView = function () {
    var v1 = document.getElementById("v1");
    var v2 = document.getElementById("v2");
    var v3 = document.getElementById("v3");
    if (multiView) {
        multiView = false;
        // hide the extra videos
        v1.style.display = "none";
        v2.style.display = "none";
        v3.style.display = "none";
        // delete the stream
        while (videoStreams.length > 1) {
            const stream = videoStreams.pop();
            stream.Destroy();
        }
    } else {
        multiView = true;
        // display the extra videos
        v1.style.display = "inline-block";
        v2.style.display = "inline-block";
        v3.style.display = "inline-block";
        // create the stream
        videoStreams.push(new Stream("SliceFrontStream", document.getElementById("v1"), videoClient));
        videoStreams.push(new Stream("SliceRightStream", document.getElementById("v2"), videoClient));
        videoStreams.push(new Stream("SliceTopStream", document.getElementById("v3"), videoClient));
    }
    setVideoSize();
    updateView();
};

/**
 * Change Angle handler.
 *
 * @param {event} event
 */
window.onChangeAngle = function (event) {
    viewAngle = parseInt(event.target.value);
    updateCamera();
}

/**
 * Change transfer function min handler.
 *
 * @param {event} event
 */
window.onChangeTFMin = function (event) {
    transferFunctionMin = parseInt(event.target.value);
    if (transferFunctionMin >= transferFunctionMax) {
        transferFunctionMax = transferFunctionMin + 1;
    }
    updateTransferFunction();
}

/**
 * Change transfer function max handler.
 *
 * @param {event} event
 */
window.onChangeTFMax = function (event) {
    transferFunctionMax = parseInt(event.target.value);
    if (transferFunctionMax <= transferFunctionMin) {
        transferFunctionMin = transferFunctionMax - 1;
    }
    updateTransferFunction();
}

/**
 * Start.
 */
window.start = function () {
    // first reset the RenderServer to make sure no old settings are live
    var resetRequest = new proto.renderServer.ResetRequest();
    resetRequest.addInterfaces(proto.renderServer.ResetRequest.Interface.ALL);
    grpcClosureRSSync(resetRequest, "reset");

    var postProcessDenoiseRequest = new proto.renderServer.PostProcessDenoiseRequest();
    postProcessDenoiseRequest.setMethod(proto.renderServer.PostProcessDenoiseRequest.Method.AI);
    postProcessDenoiseRequest.setEnableIterationLimit(proto.types.Switch.SWITCH_ENABLE);
    postProcessDenoiseRequest.setIterationLimit(500);
    grpcClosureRSAsync(postProcessDenoiseRequest, "postProcessDenoise");

    var renderSettingsRequest = new proto.renderServer.RenderSettingsRequest();
    // set the interation count, the higher the better the image quality but this also increases render time
    renderSettingsRequest.setMaxIterations(1000);
    grpcClosureRSAsync(renderSettingsRequest, "renderSettings");

    // set the background light
    var backgroundLightRequest = new proto.renderServer.BackgroundLightRequest();

    backgroundLightRequest.setIntensity(0.5);
    var color = new proto.types.Float3();
    color.setX(1.0);
    color.setY(1.0);
    color.setZ(1.0);
    backgroundLightRequest.setTopColor(color);
    backgroundLightRequest.setHorizonColor(color);
    backgroundLightRequest.setBottomColor(color);
    backgroundLightRequest.setEnable(proto.types.Switch.SWITCH_ENABLE);
    backgroundLightRequest.setCastLight(proto.types.Switch.SWITCH_ENABLE);
    backgroundLightRequest.setShow(proto.types.Switch.SWITCH_ENABLE);
    grpcClosureRSAsync(backgroundLightRequest, "backgroundLight");

    // start the video streams
    videoStreams.push(new Stream("CinematicStream", document.getElementById("v0"), videoClient));
    window.addEventListener("resize", (event) => {
        setVideoSize();
    });
    setVideoSize();

    updateCamera();
    updateView();
}
