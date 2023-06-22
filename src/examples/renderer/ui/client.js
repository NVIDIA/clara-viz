/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

const grpc = {};
grpc.web = require("grpc-web");

const {
    RenderServerClient,
} = require("./examples/renderer/example_render_server_grpc_web_pb.js");
const { VideoClient } = require("./nvidia/claraviz/video/v1/video_grpc_web_pb.js");

const proto = {};
proto.renderer = require("./examples/renderer/example_render_server_pb.js");
proto.camera = require("./nvidia/claraviz/core/camera_pb.js");
proto.types = require("./nvidia/claraviz/core/types_pb.js");
proto.video = require("./nvidia/claraviz/video/v1/video_pb.js");

const renderServerHost =
    window.location.protocol + "//" + window.location.hostname + ":8082";

var rsClient = new RenderServerClient(renderServerHost);
var videoClient = new VideoClient(renderServerHost);

//////////////////////////////
// Camera animation parameters
//////////////////////////////
// spherical coordinates
var viewTheta = 0.0;
var viewPhi = 0.0;
// distance from origin
const CAMERA_DISTANCE = 3.0;
// rotation speed
const CAMERA_ROTATION_SPEED_THETA = 0.3;
const CAMERA_ROTATION_SPEED_PHI = 0.5;
// field of view
const CAMERA_FOV = 35.0;
// frames per second
const VIDEO_FPS = 30.0;

//////////////////////////////
// Slice animation parameters
//////////////////////////////
// current slices
var sliceX = 0.0;
var sliceY = 0.0;
var sliceZ = 0.0;
// slice change speed
const SLICE_SPEED_X = 0.3;
const SLICE_SPEED_Y = 0.1;
const SLICE_SPEED_Z = 0.2;

function runAnimation() {
    // rotate the camera
    var cameraRequest = new proto.camera.CameraRequest();

    var eye = new proto.types.Float3();
    eye.setX(CAMERA_DISTANCE * Math.sin(viewTheta) * Math.cos(viewPhi));
    eye.setY(CAMERA_DISTANCE * Math.sin(viewTheta) * Math.sin(viewPhi));
    eye.setZ(CAMERA_DISTANCE * Math.cos(viewTheta));
    cameraRequest.setEye(eye);
    var lookAt = new proto.types.Float3();
    lookAt.setX(0.0);
    lookAt.setY(0.0);
    lookAt.setZ(0.0);
    cameraRequest.setLookAt(lookAt);
    var up = new proto.types.Float3();
    up.setX(0.0);
    up.setY(-1.0);
    up.setZ(0.0);
    cameraRequest.setUp(up);
    cameraRequest.setFieldOfView(CAMERA_FOV);
    cameraRequest.setPixelAspectRatio(1.0);

    viewTheta += (CAMERA_ROTATION_SPEED_THETA * (2 * Math.PI)) / VIDEO_FPS;
    viewPhi += (CAMERA_ROTATION_SPEED_PHI * (2 * Math.PI)) / VIDEO_FPS;

    rsClient.camera(cameraRequest, {}, (err, response) => {
        if (err) {
            console.log(err.code);
            console.log(err.message);
        }
    });

    // move through slices
    var sliceRequest = new proto.renderer.SliceRequest();

    var slice = new proto.types.Float3();
    slice.setX(sliceX);
    slice.setY(sliceY);
    slice.setZ(sliceZ);
    sliceRequest.setSlice(slice);

    sliceX += SLICE_SPEED_X / VIDEO_FPS;
    if (sliceX > 1.0) {
        sliceX = 0.0;
    }
    sliceY += SLICE_SPEED_Y / VIDEO_FPS;
    if (sliceY > 1.0) {
        sliceY = 0.0;
    }
    sliceZ += SLICE_SPEED_Z / VIDEO_FPS;
    if (sliceZ > 1.0) {
        sliceZ = 0.0;
    }

    rsClient.slice(sliceRequest, {}, (err, response) => {
        if (err) {
            console.log(err.code);
            console.log(err.message);
        }
    });
}

async function configVideo(width, height) {
    // size needs to be > 64 and evenly divisible by 2
    const newWidth = Math.max(64, width & ~1);
    const newHeight = Math.max(64, height & ~1);
    // the bitrate depends on the resolution and the fps
    const bitRate = newWidth * newHeight * 4 * (VIDEO_FPS / 60.0 + 0.5);

    console.log(
        "Resizing the video to ",
        newWidth,
        newHeight,
        "@",
        bitRate / (1024 * 1024.0),
        "Mbps"
    );

    // configure the video stream
    var configRequest = new proto.video.ConfigRequest();
    configRequest.setWidth(newWidth);
    configRequest.setHeight(newHeight);
    configRequest.setFrameRate(VIDEO_FPS);
    configRequest.setBitRate(bitRate);
    await videoClient.config(configRequest, {}, (err, response) => {
        if (err) {
            console.log(err.code);
            console.log(err.message);
        } else {
            console.log("Video config request success");
        }
    });
}

// video codec
const CODEC = 'video/mp4;codecs="avc1.64001F"';

var mediaSource;
var videoElement;
var sourceBuffer;
var streamOpen;
var bufArray;
var arraySize;

function createMediaSource() {
    if (!window.MediaSource) {
        console.error("No Media Source API available");
        return;
    }
    if (!MediaSource.isTypeSupported(CODEC)) {
        console.log("codec not supported");
        return;
    }

    streamOpen = false;
    bufArray = new Array();
    arraySize = 0;

    mediaSource = new MediaSource();

    videoElement = document.getElementById("v");
    videoElement.src = window.URL.createObjectURL(mediaSource);
    // reaload after changing the source
    videoElement.load();

    mediaSource.addEventListener("sourceopen", () => {
        // Setting duration to `Infinity` makes video behave like a live stream.
        mediaSource.duration = +Infinity;
        streamOpen = true;
        sourceBuffer = mediaSource.addSourceBuffer(CODEC);
        // Segments are appended to the SourceBuffer in a strict sequence
        sourceBuffer.mode = "sequence";
        sourceBuffer.addEventListener("updateend", () => {
            flushBufferedFrames();
        });
    });

    mediaSource.addEventListener("sourceclose", () => {
        streamOpen = false;
    });

    mediaSource.addEventListener("sourceended", () => {
        streamOpen = false;
    });
}

function flushBufferedFrames() {
    if (arraySize && streamOpen && !sourceBuffer.updating) {
        // combine the buffers
        var streamBuffer = new Uint8Array(arraySize);
        let i = 0;
        while (bufArray.length > 0) {
            var buf = bufArray.shift();
            streamBuffer.set(buf, i);
            i += buf.length;
        }

        // add the received data to the source buffer
        sourceBuffer.appendBuffer(streamBuffer);

        arraySize = 0;
    }
}

function decodeAndDisplayVideo(buffer) {
    // if the stram is open and the source buffer object is not updating
    // we can append the buffer
    if (streamOpen && !sourceBuffer.updating) {
        if (arraySize) {
            // if there are already buffer frames append the current frame
            bufArray.push(buffer);
            arraySize += buffer.length;
            // and flush the buffered frames
            flushBufferedFrames();
        } else {
            // else directly append to the source buffer
            sourceBuffer.appendBuffer(buffer);
        }
    } else {
        // buffer the frame until the source buffer is ready again
        if (buffer !== null) {
            bufArray.push(buffer);
            arraySize += buffer.length;
        }
    }
}

async function runVideo() {
    // configure the video with the inital window size
    var videoDivElement = document.getElementById("v_div");
    configVideo(videoDivElement.clientWidth, videoDivElement.clientHeight);

    // connect the video stream
    var streamRequest = new proto.video.StreamRequest();
    var stream = videoClient.stream(streamRequest);

    stream.on("data", (response) => {
        console.log("Video stream response");

        if (response.getNewStream()) {
            console.log("New video stream");
            // create the media source object
            createMediaSource();
        }

        decodeAndDisplayVideo(response.getData_asU8());
    });

    stream.on("status", (status) => {
        console.log("Video status");
        if (status.code != grpc.web.StatusCode.OK) {
            console.log(status.code);
            console.log(status.details);
        }
        if (status.metadata) {
            console.log("Received metadata");
            console.log(status.metadata);
        }
    });

    stream.on("error", (err) => {
        console.log(err.code);
        console.log(err.message);
    });

    stream.on("end", () => {
        console.log("Video end");
        if (streamOpen && !sourceBuffer.updating) {
            mediaSource.endOfStream();
        }
    });

    // make sure to close the stream before the page is unload
    window.addEventListener("beforeunload", () => {
        // stop playing
        var controlRequest = new proto.video.ControlRequest();
        controlRequest.setState(proto.video.ControlRequest.State.STOP);
        videoClient.control(controlRequest, {}, (err, response) => {
            if (err) {
                console.log(err.code);
                console.log(err.message);
            } else {
                console.log("Video control request to STOP success");
            }
        });
        // cancel the stream
        if (stream) {
            console.log("canceling stream...");
            stream.cancel();
            stream = null;
        }
        mediaSource = null;
    });

    window.addEventListener("resize", (event) => {
        // resize the video
        configVideo(window.innerWidth, window.innerHeight);
    });

    // start playing
    var controlRequest = new proto.video.ControlRequest();
    controlRequest.setState(proto.video.ControlRequest.State.PLAY);
    await videoClient.control(controlRequest, {}, (err, response) => {
        if (err) {
            console.log(err.code);
            console.log(err.message);
        } else {
            console.log("Video control request to PLAY success");
        }
    });
}

var animated = false;
var cameraInterval;

window.toggleAnimation = function () {
    if (animated) {
        // stop the animation
        clearInterval(cameraInterval);
        animated = false;
    } else {
        // start the animation (interval time is ms, call once per frame)
        cameraInterval = setInterval(runAnimation, 1000 / VIDEO_FPS);
        animated = true;
    }
};

if (require.main === module) {
    runVideo();
}
