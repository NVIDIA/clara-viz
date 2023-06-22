/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
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

import Video from "./../../../nvidia/claraviz/video/v1/video_pb.js";

/**
 * Class handling a video stream, feeding it into a MediaSource and display it with a HTML video element.
 */
class Stream {
    videoElement = null;

    videoClient = null;

    stream = null;

    width = 0;
    height = 0;

    sourceBuffer = null;
    mediaSource = null;
    bufArray = null;
    streamOpen = false;
    sourceBuffer = null;

    /**
     * Construct
     *
     * @param {*} videoElement HTML video element
     * @param {*} videoClient gRPC video client
     */
    constructor(videoElement, videoClient) {
        this.videoElement = videoElement;
        this.videoClient = videoClient;

        // initialize the MediaSource
        if (!window.MediaSource) {
            console.log("No Media Source API available");
            return;
        }

        const mimecodec = 'video/mp4;codecs="avc1.64001F"';
        if (!MediaSource.isTypeSupported(mimecodec)) {
            console.log("codec not supported");
            return;
        }

        this.bufArray = new Array();
        this.streamOpen = false;

        this.mediaSource = new MediaSource();
        this.videoElement.src = window.URL.createObjectURL(this.mediaSource);
        // reload after changing the source
        this.videoElement.load();

        this.mediaSource.addEventListener("sourceopen", () => {
            // Setting duration to `Infinity` makes video behave like a live stream.
            this.mediaSource.duration = +Infinity;
            this.sourceBuffer = this.mediaSource.addSourceBuffer(mimecodec);
            // Segments are appended to the SourceBuffer in a strict sequence
            this.sourceBuffer.mode = "sequence";
            this.streamOpen = true;
            this.sourceBuffer.addEventListener("updateend", () => {
                // After end of update try to append pending data (if any)
                this.#decodeAndDisplayVideo(null);
            });
        });

        this.mediaSource.addEventListener("sourceclose", () => {
            this.streamOpen = false;
            this.sourceBuffer = null;
        });

        this.mediaSource.addEventListener("sourceended", () => {
            this.streamOpen = false;
            this.sourceBuffer = null;
        });

        // start the stream
        this.#videoStream();

        // stop video on visibility change (this will start to play the stream when
        // the window is currently visible)
        document.addEventListener("visibilitychange", () => { this.#visibilityChange() });
        this.#visibilityChange();

        // make sure to close the stream before the page is unload
        window.addEventListener("beforeunload", () => {
            // stop playing
            this.#videoControl(Video.ControlRequest.State.STOP);
            // cancel the stream
            if (this.stream) {
                this.stream.cancel();
            }
        });
    }

    /**
     * Sets the client width.
     *
     * @param {number} width
     * @param {number} height
     */
    ClientSize(width, height) {
        this.#videoConfig(
            Math.floor(width * window.devicePixelRatio),
            Math.floor(window.devicePixelRatio * height)
        );
    }

    /**
     * Called when the visibility of the browser change.
     */
    #visibilityChange = () => {
        // start/stop video depending on visiblity
        if (document.hidden) {
            this.#videoControl(Video.ControlRequest.State.PAUSE);
        } else {
            this.#videoControl(Video.ControlRequest.State.PLAY);
        }
    };

    /**
     * Send a gRPC request and wait for the result.
     *
     * @param {*} request to send
     * @param {string} key of the function to call
     */
    #grpcClosure = async (request, key) => {
        return await this.videoClient[key](request, {}, (err, response) => {
            if (err) {
                console.log("gRPC request ", key, " failed");
                console.log(err.code);
                console.log(err.message);
                console.log(response);
            }
        });
    };

    /**
     * Configure the video
     *
     * @param {number} width
     * @param {number} height
     */
    #videoConfig = (width, height) => {
        if (width != this.width || height != this.height) {
            this.width = width;
            this.height = height;

            const req = new Video.ConfigRequest();
            // size has to be a multiple of 2
            req.setWidth(this.width + (this.width % 2));
            req.setHeight(this.height + (this.height % 2));
            const fps = 30.0;
            req.setFrameRate(fps);
            // use the 'Kush Gauge' to estimate the bit rate
            const motionFactor = 1; // (1 - low, 2 - medium, 4 - high)
            const bitrate = width * height * fps * motionFactor * 0.07;
            req.setBitRate(Math.ceil(bitrate));
            this.#grpcClosure(req, "config");
        }
    };

    /**
     * Send a video control request.
     *
     * @param {Video.ControlRequest.State} state
     */
    #videoControl = (state) => {
        const req = new Video.ControlRequest();
        req.setState(state);
        this.#grpcClosure(req, "control");
    };

    /**
     * Starte the video stream
     */
    #videoStream = () => {
        const req = new Video.StreamRequest();
        this.stream = this.videoClient.stream(req, {}, (err, response) => {
            if (err) {
                console.log("gRPC request ", key, " failed");
                console.log(err.code);
                console.log(err.message);
                console.log(response);
            }
        });

        this.stream
            .on("data", (response) => {
                this.#decodeAndDisplayVideo(
                    response.getData_asU8()
                );
            })
            .on("status", (status) => {
                console.log("Stream status: ");
                console.log(status);
            })
            .on("error", (err) => {
                console.log("Stream error: ");
                console.log(err);
            })
            .on("end", () => {
                console.log("Stream end");
                if (this.streamOpen && !this.sourceBuffer.updating) {
                    this.mediaSource.endOfStream();
                }
                this.mediaSource = null;
                this.sourceBuffer = null;
                this.streamOpen = false;
            });
    };

    /**
     * Decode and display a video buffer.
     *
     * @param {} buffer
     */
    #decodeAndDisplayVideo = (buffer) => {
        if (this.sourceBuffer === null) {
            return;
        }

        if (buffer !== null) {
            const bs = new Uint8Array(buffer.slice());
            this.bufArray.push(bs);
        }

        if (
            this.streamOpen &&
            !this.sourceBuffer.updating &&
            this.bufArray.length > 0
        ) {
            // Add the received data to the source buffer. If there is an 'QuotaExceededError' try appending
            // smaller fragments. If that fails abort the current segment if active and remove all data until
            // two seconds before.
            // See https://developers.google.com/web/updates/2017/10/quotaexceedederror.
            var data = this.bufArray.shift();
            const original_length = data.length;
            var piece_length = data.length;
            var aborted = false;
            while (data.length !== 0) {
                if (this.sourceBuffer.updating) {
                    break;
                }
                try {
                    this.sourceBuffer.appendBuffer(data.slice(0, piece_length));
                    data = data.slice(piece_length);
                    piece_length = data.length;
                } catch (e) {
                    if (e.name !== "QuotaExceededError") {
                        aborted = true;
                        break;
                    }

                    // Reduction the size and try again, stop when retried too many times
                    piece_length = Math.ceil(piece_length * 0.8);
                    if (piece_length / original_length < 0.04) {
                        aborted = true;
                        break;
                    }
                }
            }
            if (aborted) {
                if (this.sourceBuffer.updating) {
                    this.sourceBuffer.abort();
                }
                const currentTime = this.videoElement.currentTime;
                if (currentTime > 2) {
                    this.sourceBuffer.remove(0, currentTime - 2);
                }
            }
            // add any data we could not append above back to the buffer array
            if (data.length !== 0) {
                this.bufArray.unshift(data);
            }
        }
    };
}

export default Stream;
