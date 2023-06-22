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

import React, { Component } from "react";

import Stream from "./source/Stream.js";
import MouseInteraction from "./../mouseInteraction/MouseInteraction.js";

class VideoStream extends Component {
    stream = null;
    interaction = null;

    videoRef = React.createRef();

    componentDidMount() {
        this.stream = new Stream(this.videoRef.current, this.props.videoClient);
        this.interaction = new MouseInteraction(
            this.videoRef.current,
            this.props.renderClient
        );

        // set the stream size to the size of the client area
        this.stream.ClientSize(
            this.videoRef.current.clientWidth,
            this.videoRef.current.clientHeight
        );

        window.addEventListener("resize", () => {
            // update the size of the stream
            this.stream.ClientSize(
                this.videoRef.current.clientWidth,
                this.videoRef.current.clientHeight
            );
        });
    }

    render() {
        return (
            <video
                style={{
                    position: "fixed",
                    top: "0",
                    left: "0",
                    width: "100%",
                    height: "100%",
                }}
                autoPlay
                muted
                width="auto"
                height="auto"
                ref={this.videoRef}
            />
        );
    }
}

export default VideoStream;
