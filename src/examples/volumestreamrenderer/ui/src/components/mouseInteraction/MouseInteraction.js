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

import { Vector2, Vector3 } from "three";

import Renderer from "./../../volume_stream_render_server_pb.js";
import Camera from "./../../nvidia/claraviz/core/camera_pb.js";
import Types from "./../../nvidia/claraviz/core/types_pb.js";

class TouchedPoint {
    constructor(pointerId, x, y) {
        this.pointerId = pointerId;
        this.x = x;
        this.y = y;
        this.start_x = x;
        this.start_y = y;
    }
}

const _newVector = (obj) => new Vector3(obj.x, obj.y, obj.z);

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

class MouseInteraction {
    element = null;
    renderClient = null;

    touchedPoints = [];
    key = "";
    multiTouchDistance = 0;
    multiTouchCenter = new Vector2(0, 0);
    button = -1;

    datasetInfo = {
        size: { x: 1, y: 1, z: 1 },
        elementSize: { x: 1.0, y: 1.0, z: 1.0 },
    };

    settings = {
        Camera: {
            name: "Cinematic",
            eye: { x: 0.438, y: 0.193, z: 0.94 },
            lookAt: { x: 0.0, y: 0.0, z: 0.0 },
            up: { x: 0.0, y: 1.0, z: 0.0 },
            fieldOfView: 30,
            pixelAspectRatio: 1,
        },
        View: {
            mode: "CINEMATIC",
            cameraName: "Cinematic",
            dataViewName: "DataView",
        },
        DataView: {
            name: "DataView",
            zoomFactor: 1.0,
            viewOffset: { x: 0.0, y: 0.0 },
            dataOffset: [0.0, 0.0, 0.0],
            pixelAspectRatio: 1.0,
        },
    };

    constructor(element, renderClient) {
        this.element = element;
        this.renderClient = renderClient;

        element.addEventListener("pointermove", this.onPointerMove);
        element.addEventListener("pointerdown", this.onPointerDown);
        element.addEventListener("pointerup", this.onPointerUp);
        element.addEventListener("pointercancel", this.onPointerCancel);
        element.addEventListener("pointerout", this.onPointerOut);
        element.addEventListener("wheel", this.onWheel);
    }

    grpcClosure = async (request, key) => {
        return await this.renderClient[key](request, {}, (err, response) => {
            if (err) {
                console.log("gRPC request ", key, " failed");
                console.log(err.code);
                console.log(err.message);
                console.log(response);
            }
        });
    };

    updateMultiTouch = () => {
        const dx = this.touchedPoints[1].x - this.touchedPoints[0].x;
        const dy = this.touchedPoints[1].y - this.touchedPoints[0].y;
        this.multiTouchDistance = Math.sqrt(dx * dx + dy * dy);

        this.multiTouchCenter.x = this.touchedPoints[0].x + dx * 0.5;
        this.multiTouchCenter.y = this.touchedPoints[0].y + dy * 0.5;
    };

    handleCameraUpdate = () => {
        var req = new Camera.CameraRequest();
        req.setName(this.settings.Camera.name);
        req.setEye(
            new Types.Float3([
                this.settings.Camera.eye.x,
                this.settings.Camera.eye.y,
                this.settings.Camera.eye.z,
            ])
        );
        req.setLookAt(
            new Types.Float3([
                this.settings.Camera.lookAt.x,
                this.settings.Camera.lookAt.y,
                this.settings.Camera.lookAt.z,
            ])
        );
        req.setUp(
            new Types.Float3([
                this.settings.Camera.up.x,
                this.settings.Camera.up.y,
                this.settings.Camera.up.z,
            ])
        );
        req.setFieldOfView(this.settings.Camera.fieldOfView);
        req.setPixelAspectRatio(this.settings.Camera.pixelAspectRatio);

        this.grpcClosure(req, "camera");
    };

    handleDataViewUpdate = () => {
        //const dataviewRequest = Renderer.DataViewRequest.fromObject(this.settings.DataView);
        //this.grpcClosure(req, "dataview");
    };

    getScreenSize = () => {
        const screen_size = { screen_width_mm: 1, screen_height_mm: 1 };

        if (typeof this.settings.View === "undefined") {
            return screen_size;
        }

        const { View } = this.settings;

        if (View.mode === "TWOD") {
            if (
                typeof this.settings.DataView === "undefined" ||
                typeof this.datasetInfo === "undefined"
            ) {
                return screen_size;
            }

            const { DataView } = this.settings;

            if (this.element.clientWidth > this.element.clientHeight) {
                screen_size.screen_height_mm =
                    (this.datasetInfo.elementSize.y * this.datasetInfo.size.y) /
                    DataView.zoomFactor;
                screen_size.screen_width_mm =
                    screen_size.screen_height_mm *
                    (this.element.clientWidth / this.element.clientHeight);
            } else {
                screen_size.screen_width_mm =
                    (this.datasetInfo.elementSize.x * this.datasetInfo.size.x) /
                    DataView.zoomFactor;
                screen_size.screen_height_mm =
                    screen_size.screen_width_mm *
                    (this.element.clientHeight / this.element.clientWidth);
            }
        } else if (
            View.mode === "SLICE" ||
            View.mode === "SLICE_SEGMENTATION"
        ) {
            if (typeof this.settings.Camera === "undefined") {
                return screen_size;
            }

            const { Camera } = this.settings;

            // calculate the viewport
            const aspect_ratio =
                this.element.clientHeight / this.element.clientWidth;
            const half_view_size = Math.tan(
                (0.5 * Camera.fieldOfView * Math.PI) / 180.0
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
            const nz = _newVector(Camera.lookAt).sub(_newVector(Camera.eye));
            const target_distance = nz.length();
            nz.normalize();
            const nx = _newVector(Camera.up).cross(nz);
            nx.normalize();
            const ny = nz.clone().cross(nx);
            ny.normalize();

            // calculate the left/top, left/bottom and right/top positions in world space
            const left_top = _newVector(Camera.eye)
                .add(nx.clone().multiplyScalar(left_view))
                .add(ny.clone().multiplyScalar(top_view))
                .add(nz.clone().multiplyScalar(target_distance));
            const left_bottom = _newVector(Camera.eye)
                .add(nx.clone().multiplyScalar(left_view))
                .add(ny.clone().multiplyScalar(top_view + height_view))
                .add(nz.clone().multiplyScalar(target_distance));
            const right_top = _newVector(Camera.eye)
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
    };

    pan = (dx, dy) => {
        if (typeof this.settings.View === "undefined") {
            return;
        }
        const { View } = this.settings;

        if (View.mode === "TWOD") {
            if (typeof this.settings.DataView === "undefined") {
                return;
            }

            const { screen_width_mm, screen_height_mm } = this.getScreenSize();

            const { DataView } = this.settings;
            DataView.viewOffset.x -=
                (dx / this.element.clientWidth) * screen_width_mm;
            DataView.viewOffset.y -=
                (dy / this.element.clientHeight) * screen_height_mm;

            this.handleDataViewUpdate();
        } else {
            if (typeof this.settings.Camera === "undefined") {
                return;
            }

            const { Camera } = this.settings;

            var eye = _newVector(Camera.eye),
                lookAt = _newVector(Camera.lookAt),
                up = _newVector(Camera.up),
                vec = lookAt.clone().sub(eye),
                right = up.clone().cross(vec);

            var length;
            if (this.settings.View.cameraName.startsWith("Slice")) {
                // slice mode is using ortho projection where the distance of the eye to the
                // look-at position should not affect the pan distance
                length = 1.0;
            } else {
                length = vec.length();
            }

            right.normalize();
            const horiz = length * (dx / this.element.clientWidth),
                vert = length * (-dy / this.element.clientHeight);

            right.multiplyScalar(horiz);
            const nup = up.clone().multiplyScalar(vert);
            eye.add(right).sub(nup);
            lookAt.add(right).sub(nup);

            Camera.eye = Object.assign({}, eye);
            Camera.lookAt = Object.assign({}, lookAt);

            this.handleCameraUpdate();
        }
    };

    orbit = (dx, dy) => {
        if (
            typeof this.settings.Camera === "undefined" ||
            typeof this.settings.View === "undefined"
        ) {
            return;
        }

        const { Camera } = this.settings;

        var eye = _newVector(Camera.eye);

        const rightDegree = (dx / 180.0) * Math.PI,
            downDegree = (dy / 180.0) * Math.PI,
            lookAt = _newVector(Camera.lookAt),
            up = _newVector(Camera.up),
            yup = _newVector(Camera.up),
            vec = eye.clone().sub(lookAt),
            right = up.clone().cross(vec);

        _rotateAxis(vec, right, downDegree);
        _rotateAxis(vec, yup, rightDegree);

        _rotateAxis(up, right, downDegree);
        _rotateAxis(up, yup, rightDegree);

        eye = vec.clone().add(lookAt);

        Camera.eye = Object.assign({}, eye);
        Camera.up = Object.assign({}, up);

        this.handleCameraUpdate();
    };

    zoom = (factor) => {
        if (typeof this.settings.View === "undefined") {
            return;
        }
        const { View } = this.settings;

        if (View.mode === "TWOD") {
            if (typeof this.settings.DataView === "undefined") {
                return;
            }

            const { DataView } = this.settings;

            if (factor < 1.0 || DataView.zoomFactor > 1.0) {
                DataView.zoomFactor = Math.max(
                    DataView.zoomFactor / factor,
                    1.0
                );
                this.handleDataViewUpdate();
            }
        } else {
            if (typeof this.settings.Camera === "undefined") {
                return;
            }
            const { Camera, View } = this.settings;

            if (View.mode === "CINEMATIC") {
                var view_vec = _newVector(Camera.eye);
                view_vec.sub(Camera.lookAt);

                if (factor > 1.0 || view_vec.length() > 0.0005) {
                    view_vec.multiplyScalar(factor);
                    view_vec.add(Camera.lookAt);
                    Camera.eye = Object.assign({}, view_vec);
                    this.handleCameraUpdate();
                }
            } else {
                if (factor > 1.0 || Camera.fieldOfView > 0.1) {
                    Camera.fieldOfView *= factor;
                    this.handleCameraUpdate();
                }
            }
        }
    };

    slice = (dx, dy) => {
        if (
            typeof this.settings.Camera === "undefined" ||
            typeof this.datasetInfo === "undefined"
        ) {
            return;
        }

        const { Camera } = this.settings;

        // volume element size is in millimeters, volume size in meters
        const element_size = _newVector(
            this.datasetInfo.elementSize
        ).multiplyScalar(0.001);
        const volume_size = _newVector(this.datasetInfo.size).multiply(
            element_size
        );
        const volume_min = _newVector(volume_size).multiplyScalar(-0.5);
        const volume_max = _newVector(volume_size).multiplyScalar(0.5);

        const view_vec = _newVector(Camera.eye).sub(Camera.lookAt).normalize();

        const delta = _newVector(view_vec)
            .multiply(volume_size)
            .multiplyScalar(dy / this.element.clientHeight);

        var new_lookAt = _newVector(Camera.lookAt).add(delta);
        var new_eye = _newVector(Camera.eye).add(delta);

        // calculate the distance of the volume to the new look-at point
        const dist0 = _newVector(volume_min).sub(new_lookAt).dot(view_vec);
        const dist1 = new Vector3(volume_min.x, volume_min.y, volume_max.z)
            .sub(new_lookAt)
            .dot(view_vec);
        const dist2 = new Vector3(volume_min.x, volume_max.y, volume_min.z)
            .sub(new_lookAt)
            .dot(view_vec);
        const dist3 = new Vector3(volume_min.x, volume_max.y, volume_max.z)
            .sub(new_lookAt)
            .dot(view_vec);
        const dist4 = new Vector3(volume_max.x, volume_min.y, volume_min.z)
            .sub(new_lookAt)
            .dot(view_vec);
        const dist5 = new Vector3(volume_max.x, volume_min.y, volume_max.z)
            .sub(new_lookAt)
            .dot(view_vec);
        const dist6 = new Vector3(volume_max.x, volume_max.y, volume_min.z)
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
            Camera.lookAt = Object.assign({}, new_lookAt);
            Camera.eye = Object.assign({}, new_eye);
            this.handleCameraUpdate();
        }
    };

    onPointerMove = (e) => {
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
                if (typeof this.settings.View !== "undefined") {
                    if (
                        this.settings.View.mode === "CINEMATIC" ||
                        (typeof this.settings.View.cameraName !== "undefined" &&
                            this.settings.View.cameraName.includes("Oblique") &&
                            shiftKey)
                    ) {
                        this.orbit(dx, dy);
                    } else if (this.settings.View.mode !== "TWOD") {
                        this.slice(dx, dy);
                    }
                }
            }

            this.touchedPoints[0].x = clientX;
            this.touchedPoints[0].y = clientY;
        }
    };

    onPointerDown = (e) => {
        const { clientX, clientY, button, pointerId } = e;

        this.touchedPoints.push(new TouchedPoint(pointerId, clientX, clientY));

        // make sure the 'clara-measure' receives events for this pointer ID
        this.element.setPointerCapture(pointerId);

        // handle transition to single or multi touch
        if (this.touchedPoints.length === 2) {
            updateMultiTouch();
        } else if (this.touchedPoints.length === 1) {
            this.button = button;
        }

        e.preventDefault();
    };

    onPointerUp = (e) => {
        const { button, pointerId } = e;

        // release the pointer capture for 'clara-interaction'
        this.element.releasePointerCapture(pointerId);

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
    };

    onWheel = (e) => {
        const { deltaY } = e;

        this.zoom(deltaY > 0 ? 1.05 : 0.95);

        e.preventDefault();
        e.stopPropagation();
        this.element.focus({ preventScroll: true });
    };

    onPointerOut = (e) => {
        const { pointerType } = e;

        if (pointerType === "mouse") {
            this.onPointerUp(e);
        }
    };

    onPointerCancel = (e) => {
        this.onPointerUp(e);
    };
}

export default MouseInteraction;
