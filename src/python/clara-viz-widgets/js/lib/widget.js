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

var widgets = require("@jupyter-widgets/base");
var _ = require("lodash");
var THREE = require("three");

var semver_range = require("../package.json").version

// See widget.py for the kernel counterpart to this file.

var WidgetModel = widgets.DOMWidgetModel.extend({
    defaults: _.extend(widgets.DOMWidgetModel.prototype.defaults(), {
        _model_name: "WidgetModel",
        _view_name: "WidgetView",
        _model_module: "clara-viz-widgets",
        _view_module: "clara-viz-widgets",
        _model_module_version: semver_range,
        _view_module_version: semver_range,
    }),
    nviews: 0,
});

var TouchedPoint = class {
    constructor(pointerId, x, y) {
        this.pointerId = pointerId;
        this.x = x;
        this.y = y;
        this.start_x = x;
        this.start_y = y;
    }
};

// Custom View. Renders the widget model.
var WidgetView = widgets.DOMWidgetView.extend({
    video: null,
    sourceBuffer: null,
    mediaSource: null,
    bufArray: null,
    streamOpen: false,
    viewId: null,
    intersectionObserver: null,
    videoIsIntersecting: false,

    touchedPoints: [],
    key: "",
    multiTouchDistance: 0,
    multiTouchCenter: new THREE.Vector2(0, 0),
    button: -1,
    datasetInfo: null,
    settings: null,

    debug: function (msg) {
        console.log(`[view ${this.viewId}]: ` + msg);
    },

    initialize: function (attributes, options) {
        WidgetView.__super__.initialize.call(this, attributes, options);

        // initialize the model change handlers
        this.model.on("change:_settings", this.settingsChange, this);
        this.settingsChange();
        this.model.on("change:_dataset_info", this.datasetInfoChange, this);
        this.datasetInfoChange();

        // connect the message handler which receives the stream data
        this.model.on("msg:custom", this.on_msg, this);

        this.viewId = this.model.nviews;
        ++this.model.nviews;
    },

    settingsChange: function () {
        this.settings = JSON.parse(this.model.get("_settings"));
    },

    datasetInfoChange: function () {
        this.datasetInfo = JSON.parse(this.model.get("_dataset_info"));
    },

    on_msg: function (msg, buffers) {
        if (msg == "stream") {
            this.decodeAndDisplayVideo(buffers[0].buffer);
        } else {
            console.log(msg);
        }
    },

    // Defines how the widget gets rendered into the DOM
    render: function () {
        var that = this;

        this.video = document.createElement("video");

        // create an intersection observer to pause the video if it's not intersecting
        // the viewport
        const options = {}
        this.intersectionObserver = new IntersectionObserver(function (entries, observer) {
            that.onIntersecting(entries, observer);
        }, options);
        this.intersectionObserver.observe(this.video);

        // also check the visibility of the browser window itself
        document.addEventListener("visibilitychange", () => {
            that.onVisibilityChange();
        });

        // add the event listeners for user interaction
        this.video.addEventListener("pointermove", function (event) {
            that.onPointerMove(event);
        });
        this.video.addEventListener("pointerdown", function (event) {
            that.onPointerDown(event);
        });
        this.video.addEventListener("pointerup", function (event) {
            that.onPointerUp(event);
        });
        this.video.addEventListener("pointercancel", function (event) {
            that.onPointerCancel(event);
        });
        this.video.addEventListener("pointerout", function (event) {
            that.onPointerOut(event);
        });
        this.video.addEventListener("wheel", function (event) {
            that.onWheel(event);
        });

        var keyDownFn = function (event) {
            that.keyDownHandler(event);
        };
        var keyUpFn = function (event) {
            that.keyUpHandler(event);
        };
        var keyPressFn = function (event) {
            that.keyPressHandler(event);
        };

        let addKeyListeners = function (e) {
            that.debug("add keyboard listeners");
            document.addEventListener("keydown", keyDownFn, true);
            document.addEventListener("keyup", keyUpFn, true);
            document.addEventListener("keypress", keyPressFn, true);
        };

        let removeKeyListeners = function (e) {
            that.debug("remove keyboard listeners");
            document.removeEventListener("keydown", keyDownFn, true);
            document.removeEventListener("keyup", keyUpFn, true);
            document.removeEventListener("keypress", keyPressFn, true);
        };

        this.video.addEventListener("focusin", addKeyListeners, false);
        this.video.addEventListener("focusout", removeKeyListeners, false);

        this.video.autoplay = true;
        this.video.muted = true;
        this.video.setAttribute("width", "auto");
        this.video.setAttribute("height", "auto");

        this.el.appendChild(this.video);

        // initialize the MediaSource
        if (!window.MediaSource) {
            that.debug("No Media Source API available");
            return;
        }

        const mimecodec = 'video/mp4;codecs="avc1.64001F"';
        if (!MediaSource.isTypeSupported(mimecodec)) {
            that.debug("codec not supported");
            return;
        }

        this.bufArray = new Array();
        this.streamOpen = false;

        this.mediaSource = new MediaSource();
        this.video.src = window.URL.createObjectURL(this.mediaSource);
        // reload after changing the source
        this.video.load();

        this.mediaSource.addEventListener("sourceopen", function () {
            // Setting duration to `Infinity` makes video behave like a live stream.
            that.mediaSource.duration = +Infinity;
            that.sourceBuffer = that.mediaSource.addSourceBuffer(mimecodec);
            // Segments are appended to the SourceBuffer in a strict sequence
            that.sourceBuffer.mode = "sequence";
            that.streamOpen = true;
            that.sourceBuffer.addEventListener("updateend", () => {
                // After end of update try to append pending data (if any)
                that.decodeAndDisplayVideo(null);
            });
        });

        this.mediaSource.addEventListener("sourceclose", function () {
            that.streamOpen = null;
        });

        this.mediaSource.addEventListener("sourceended", function () {
            that.streamOpen = null;
        });
    },

    decodeAndDisplayVideo: function (buffer) {
        if (this.sourceBuffer === null) {
            //console.log('Source buffer is empty');
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
                const currentTime = this.video.currentTime;
                if (currentTime > 2) {
                    this.sourceBuffer.remove(0, currentTime - 2);
                }
            }
            // add any data we could not append above back to the buffer array
            if (data.length !== 0) {
                this.bufArray.unshift(data);
            }
        }
    },

    updateMultiTouch: function () {
        const dx = this.touchedPoints[1].x - this.touchedPoints[0].x;
        const dy = this.touchedPoints[1].y - this.touchedPoints[0].y;
        this.multiTouchDistance = Math.sqrt(dx * dx + dy * dy);

        this.multiTouchCenter.x = this.touchedPoints[0].x + dx * 0.5;
        this.multiTouchCenter.y = this.touchedPoints[0].y + dy * 0.5;
    },

    handleCameraUpdate: function () {
        var command = {
            msg_type: "camera_update",
            contents: this.settings.Cameras,
        };
        this.send(command);
    },

    handleDataViewUpdate: function () {
        var command = {
            msg_type: "data_view_update",
            contents: this.settings.DataViews,
        };
        this.send(command);
    },

    getScreenSize: function () {
        const screen_size = { screen_width_mm: 1, screen_height_mm: 1 };

        if (typeof this.settings.Views === "undefined") {
            return screen_size;
        }

        const view = this.settings.Views[0];

        if (view.mode === "TWOD") {
            if (
                typeof this.settings.DataViews === "undefined" ||
                typeof this.datasetInfo === "undefined"
            ) {
                return screen_size;
            }

            const data_view = this.settings.DataViews.find(data_view => data_view.name === view.dataViewName);

            if (this.video.clientWidth > this.video.clientHeight) {
                screen_size.screen_height_mm =
                    (this.datasetInfo.elementSize.y * this.datasetInfo.size.y) /
                    data_view.zoomFactor;
                screen_size.screen_width_mm =
                    screen_size.screen_height_mm *
                    (this.video.clientWidth / this.video.clientHeight);
            } else {
                screen_size.screen_width_mm =
                    (this.datasetInfo.elementSize.x * this.datasetInfo.size.x) /
                    data_view.zoomFactor;
                screen_size.screen_height_mm =
                    screen_size.screen_width_mm *
                    (this.video.clientHeight / this.video.clientWidth);
            }
        } else if (
            view.mode === "SLICE" ||
            view.mode === "SLICE_SEGMENTATION"
        ) {
            if (typeof this.settings.Cameras === "undefined") {
                return screen_size;
            }

            const camera = this.settings.Cameras.find(camera => camera.name === view.cameraName);

            // calculate the viewport
            const aspect_ratio =
                this.video.clientHeight / this.video.clientWidth;
            const half_view_size = Math.tan(
                (0.5 * camera.fieldOfView * Math.PI) / 180.0
            );

            let left_view, top_view, right_view, bottom_view;
            if (aspect_ratio > 1) {
                left_view = -half_view_size;
                right_view = half_view_size;
                top_view = -half_view_size * aspect_ratio;
                bottom_view = half_view_size * aspect_ratio;
            } else {
                left_view = -half_view_size / aspect_ratio;
                right_view = half_view_size / aspect_ratio;
                top_view = -half_view_size;
                bottom_view = half_view_size;
            }
            const width_view = right_view - left_view;
            const height_view = bottom_view - top_view;

            // calculate the view vectors
            const nz = _newVector(camera.lookAt).sub(_newVector(camera.eye));
            const target_distance = nz.length();
            nz.normalize();
            const nx = _newVector(camera.up).cross(nz);
            nx.normalize();
            const ny = nz.clone().cross(nx);
            ny.normalize();

            // calculate the left/top, left/bottom and right/top positions in world space
            const left_top = _newVector(camera.eye)
                .add(nx.clone().multiplyScalar(left_view))
                .add(ny.clone().multiplyScalar(top_view))
                .add(nz.clone().multiplyScalar(target_distance));
            const left_bottom = _newVector(camera.eye)
                .add(nx.clone().multiplyScalar(left_view))
                .add(ny.clone().multiplyScalar(top_view + height_view))
                .add(nz.clone().multiplyScalar(target_distance));
            const right_top = _newVector(camera.eye)
                .add(nx.clone().multiplyScalar(left_view + width_view))
                .add(ny.clone().multiplyScalar(top_view))
                .add(nz.clone().multiplyScalar(target_distance));

            // camera setup is in meters, convert to mm by multiplying with 1000
            screen_size.screen_width_mm =
                right_top.clone().sub(left_top).length() * 1000;
            screen_size.screen_height_mm =
                left_bottom.clone().sub(left_top).length() * 1000;
        }
        return screen_size;
    },

    pan: function (dx, dy) {
        if (typeof this.settings.Views === "undefined") {
            return;
        }
        const view = this.settings.Views[0];

        if (view.mode === "TWOD") {
            if (typeof this.settings.DataViews === "undefined") {
                return;
            }

            const { screen_width_mm, screen_height_mm } = this.getScreenSize();

            const data_view = this.settings.DataViews.find(data_view => data_view.name === view.dataViewName);
            data_view.viewOffset.x -=
                (dx / this.video.clientWidth) * screen_width_mm;
            data_view.viewOffset.y -=
                (dy / this.video.clientHeight) * screen_height_mm;

            this.handleDataViewUpdate();
        } else {
            if (typeof this.settings.Cameras === "undefined") {
                return;
            }

            const camera = this.settings.Cameras.find(camera => camera.name === view.cameraName);

            var eye = _newVector(camera.eye),
                lookAt = _newVector(camera.lookAt),
                up = _newVector(camera.up),
                vec = lookAt.clone().sub(eye),
                right = up.clone().cross(vec);

            var length;
            if ((view.mode === "SLICE") || (view.mode === "SLICE_SEGMENTATION")) {
                // slice mode is using ortho projection where the distance of the eye to the
                // look-at position should not affect the pan distance
                length = camera.fieldOfView / 30.0;
            } else {
                length = vec.length();
            }

            right.normalize();
            const horiz = length * (dx / this.video.clientWidth),
                vert = length * (-dy / this.video.clientHeight);

            right.multiplyScalar(horiz);
            const nup = up.clone().multiplyScalar(vert);
            eye.add(right).sub(nup);
            lookAt.add(right).sub(nup);

            camera.eye = Object.assign({}, eye);
            camera.lookAt = Object.assign({}, lookAt);

            this.handleCameraUpdate();
        }
    },

    orbit: function (dx, dy) {
        if (
            typeof this.settings.Cameras === "undefined" ||
            typeof this.settings.Views === "undefined"
        ) {
            return;
        }

        const view = this.settings.Views[0];
        const camera = this.settings.Cameras.find(camera => camera.name === view.cameraName);

        var eye = _newVector(camera.eye);

        const rightDegree = (dx / 180.0) * Math.PI,
            downDegree = (dy / 180.0) * Math.PI,
            lookAt = _newVector(camera.lookAt),
            up = _newVector(camera.up),
            yup = _newVector(camera.up),
            vec = eye.clone().sub(lookAt),
            right = up.clone().cross(vec);

        _rotateAxis(vec, right, downDegree);
        _rotateAxis(vec, yup, rightDegree);

        _rotateAxis(up, right, downDegree);
        _rotateAxis(up, yup, rightDegree);

        eye = vec.clone().add(lookAt);

        camera.eye = Object.assign({}, eye);
        camera.up = Object.assign({}, up);

        this.handleCameraUpdate();
    },

    zoom: function (factor) {
        if (typeof this.settings.Views === "undefined") {
            return;
        }

        const view = this.settings.Views[0];

        if (view.mode === "TWOD") {
            const data_view = this.settings.DataViews.find(data_view => data_view.name === view.dataViewName);

            if (factor < 1.0 || data_view.zoomFactor > 1.0) {
                data_view.zoomFactor = Math.max(
                    data_view.zoomFactor / factor,
                    1.0
                );
                this.handleDataViewUpdate();
            }
        } else {
            const camera = this.settings.Cameras.find(camera => camera.name === view.cameraName);

            if (view.mode === "CINEMATIC") {
                var view_vec = _newVector(camera.eye);
                view_vec.sub(camera.lookAt);

                if (factor > 1.0 || view_vec.length() > 0.0005) {
                    view_vec.multiplyScalar(factor);
                    view_vec.add(camera.lookAt);
                    camera.eye = Object.assign({}, view_vec);
                    this.handleCameraUpdate();
                }
            } else {
                if (factor > 1.0 || camera.fieldOfView > 0.1) {
                    camera.fieldOfView *= factor;
                    this.handleCameraUpdate();
                }
            }
        }
    },

    slice: function (dx, dy) {
        if (typeof this.settings.Views === "undefined" ||
            typeof this.datasetInfo === "undefined") {
            return;
        }

        const view = this.settings.Views[0];
        const camera = this.settings.Cameras.find(camera => camera.name === view.cameraName);

        // volume element size is in millimeters, volume size in meters
        const element_size = _newVector(
            this.datasetInfo.elementSize
        ).multiplyScalar(0.001);
        const volume_size = _newVector(this.datasetInfo.size).multiply(
            element_size
        );
        const volume_min = _newVector(volume_size).multiplyScalar(-0.5);
        const volume_max = _newVector(volume_size).multiplyScalar(0.5);

        const view_vec = _newVector(camera.eye).sub(camera.lookAt).normalize();

        const delta = _newVector(view_vec)
            .multiply(volume_size)
            .multiplyScalar(dy / this.video.clientHeight);

        var new_lookAt = _newVector(camera.lookAt).add(delta);
        var new_eye = _newVector(camera.eye).add(delta);

        // calculate the distance of the volume to the new look-at point
        const dist0 = _newVector(volume_min).sub(new_lookAt).dot(view_vec);
        const dist1 = new THREE.Vector3(
            volume_min.x,
            volume_min.y,
            volume_max.z
        )
            .sub(new_lookAt)
            .dot(view_vec);
        const dist2 = new THREE.Vector3(
            volume_min.x,
            volume_max.y,
            volume_min.z
        )
            .sub(new_lookAt)
            .dot(view_vec);
        const dist3 = new THREE.Vector3(
            volume_min.x,
            volume_max.y,
            volume_max.z
        )
            .sub(new_lookAt)
            .dot(view_vec);
        const dist4 = new THREE.Vector3(
            volume_max.x,
            volume_min.y,
            volume_min.z
        )
            .sub(new_lookAt)
            .dot(view_vec);
        const dist5 = new THREE.Vector3(
            volume_max.x,
            volume_min.y,
            volume_max.z
        )
            .sub(new_lookAt)
            .dot(view_vec);
        const dist6 = new THREE.Vector3(
            volume_max.x,
            volume_max.y,
            volume_min.z
        )
            .sub(new_lookAt)
            .dot(view_vec);
        const dist7 = _newVector(volume_max).sub(new_lookAt).dot(view_vec);
        const min = Math.min(
            dist0,
            dist1,
            dist2,
            dist3,
            dist4,
            dist5,
            dist6,
            dist7
        );
        const max = Math.max(
            dist0,
            dist1,
            dist2,
            dist3,
            dist4,
            dist5,
            dist6,
            dist7
        );
        // only apply if look at stays within the volume bounds
        if (min * max < 0.0) {
            camera.lookAt = Object.assign({}, new_lookAt);
            camera.eye = Object.assign({}, new_eye);
            this.handleCameraUpdate();
        }
    },

    onIntersecting: function (entries, observer) {
        entries.forEach(entry => {
            if (entry.target == this.video) {
                if (this.videoIsIntersecting != entry.isIntersecting) {
                    this.videoIsIntersecting = entry.isIntersecting;
                    this.onVisibilityChange();
                }
            }
        });
    },

    /**
     * Called when the visibility of the browser changes.
     */
    onVisibilityChange() {
        try {
            // pass down video visibility, server will play/pause the video stream depending on this
            var command = {
                msg_type: "video_visible",
                contents: (document.visibilityState == "visible") && (this.videoIsIntersecting),
            };
            this.send(command);
        }
        catch (e) {
            // ignore
        }
    },

    onPointerMove: function (e) {
        const { clientX, clientY, shiftKey, pointerId } = e;

        if (this.touchedPoints.length === 2) {
            // multi-touch handling
            const prevDistance = this.multiTouchDistance;
            const prevCenter = Object.assign({}, this.multiTouchCenter);

            // update the touched point position
            var touchedPoint = this.touchedPoints.find(function (point) {
                return point.pointerId === pointerId;
            });
            if (typeof touchedPoint !== "undefined") {
                touchedPoint.x = clientX;
                touchedPoint.y = clientY;

                // calculate new distance and center
                this.updateMultiTouch();

                // zoom and pan
                if (this.multiTouchDistance !== 0.0) {
                    this.zoom(prevDistance / this.multiTouchDistance);
                }
                this.pan(
                    this.multiTouchCenter.x - prevCenter.x,
                    this.multiTouchCenter.y - prevCenter.y
                );
            }
        } else if (this.touchedPoints.length === 1) {
            // single touch and mouse handling
            const dx = clientX - this.touchedPoints[0].x;
            const dy = clientY - this.touchedPoints[0].y;

            if (this.button === 1 || (this.button === 0 && this.key === "a")) {
                // middle mouse button (or left + 'a' for Mac)
                this.pan(dx, dy);
            } else if (this.button === 0) {
                // left mouse button
                if (typeof this.settings.Views !== "undefined") {
                    const view = this.settings.Views[0];
                    if (
                        view.mode === "CINEMATIC" ||
                        (typeof view.cameraName !== "undefined" &&
                            view.cameraName.includes("Oblique") &&
                            shiftKey)
                    ) {
                        this.orbit(dx, dy);
                    } else if (view.mode !== "TWOD") {
                        this.slice(dx, dy);
                    }
                }
            }

            this.touchedPoints[0].x = clientX;
            this.touchedPoints[0].y = clientY;
        }
    },

    onPointerDown: function (e) {
        const { clientX, clientY, button, pointerId } = e;

        this.touchedPoints.push(new TouchedPoint(pointerId, clientX, clientY));

        // make sure the 'clara-measure' receives events for this pointer ID
        this.video.setPointerCapture(pointerId);

        // handle transition to single or multi touch
        if (this.touchedPoints.length === 2) {
            updateMultiTouch();
        } else if (this.touchedPoints.length === 1) {
            this.button = button;
        }

        e.preventDefault();
    },

    onPointerUp: function (e) {
        const { button, pointerId } = e;

        // release the pointer capture for 'clara-interaction'
        this.video.releasePointerCapture(pointerId);

        // remove the previously touched points
        var index = this.touchedPoints.findIndex(function (point) {
            return point.pointerId === pointerId;
        });
        if (index !== -1) {
            this.touchedPoints.splice(index, 1);
        }

        // handle transition to multi, single or no touch
        if (this.touchedPoints.length === 2) {
            this.updateMultiTouch();
        } else if (this.touchedPoints.length === 1) {
            this.button = button;
        } else if (this.touchedPoints.length === 0) {
            e.stopPropagation();
            e.preventDefault();
        }
    },

    onWheel: function (e) {
        const { deltaY } = e;

        this.zoom(deltaY > 0 ? 1.05 : 0.95);

        e.preventDefault();
        e.stopPropagation();
        this.video.focus({ preventScroll: true });
    },

    onPointerOut: function (e) {
        const { pointerType } = e;

        if (pointerType === "mouse") {
            this.onPointerUp(e);
        }
    },

    onPointerCancel: function (e) {
        this.onPointerUp(e);
    },

    // convert key event to json object
    getKeyEventJson: function (keyevent) {
        var key_json = {
            key: keyevent.key,
            keyCode: keyevent.keyCode,
            which: keyevent.which,
            charCode: keyevent.charCode,
            char: String.fromCharCode(keyevent.which),
            shiftKey: keyevent.shiftKey,
            ctrlKey: keyevent.ctrlKey,
            altKey: keyevent.altKey,
            metaKey: keyevent.metaKey,
        };

        return key_json;
    },

    // A key has been pressed (special keys)
    keyDownHandler: function (event) {
        var command = {
            msg_type: "key_down",
            contents: this.getKeyEventJson(event),
        };
        command.key_down.x = 0;
        command.key_down.y = 0;
        this.send(command);
        event.stopPropagation();
    },

    // A key has been pressed (char keys)
    keyPressHandler: function (event) {
        var command = {
            msg_type: "key_press",
            contents: this.getKeyEventJson(event),
        };
        command.key_press.x = 0;
        command.key_press.y = 0;
        this.send(command);
        event.stopPropagation();
    },

    // A key has been released (for special keys)
    keyUpHandler: function (event) {
        var command = {
            msg_type: "key_up",
            contents: this.getKeyEventJson(event),
        };
        command.key_up.x = 0;
        command.key_up.y = 0;
        this.send(command);
        event.stopPropagation();
    },
});

const _newVector = (obj) => new THREE.Vector3(obj.x, obj.y, obj.z);

const _rotateAxis = (vec, axis, angle) => {
    var cosAngle = Math.cos(angle),
        sinAngle = Math.sin(angle),
        naxis = axis.clone();

    naxis.normalize();

    var va = vec.dot(naxis),
        z = naxis.clone().multiplyScalar(va),
        x = vec.clone().sub(z),
        y = x.clone().cross(naxis);

    x.multiplyScalar(cosAngle);
    y.multiplyScalar(sinAngle);

    z.add(x).add(y);

    vec.x = z.x;
    vec.y = z.y;
    vec.z = z.z;
};

module.exports = {
    WidgetModel: WidgetModel,
    WidgetView: WidgetView,
};
