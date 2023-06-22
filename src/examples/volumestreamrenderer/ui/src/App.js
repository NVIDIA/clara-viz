/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
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
import { CircularProgress } from "@material-ui/core";

import { RenderServerClient } from "./volume_stream_render_server_grpc_web_pb.js";
import { VideoClient } from "./nvidia/claraviz/video/v1/video_grpc_web_pb.js";

import VideoStream from "./components/videostream/VideoStream";

import "./App.css";

class App extends Component {
    state = {
        renderClient: null,
        videoClient: null,
        error: null,
    };

    componentDidMount() {
        const createClient = async () => {
            const renderServerHost =
                window.location.protocol +
                "//" +
                window.location.hostname +
                ":8082";
            return {
                renderClient: new RenderServerClient(renderServerHost),
                videoClient: new VideoClient(renderServerHost),
            };
        };
        createClient()
            .then((clients) => {
                this.setState({
                    renderClient: clients.renderClient,
                    videoClient: clients.videoClient,
                });
            })
            .catch((error) => {
                this.setState({ error: error });
            });
    }

    render() {
        let content = <CircularProgress />;
        if (this.state.error) {
            content = (
                <p>Failed to connect, error {this.state.error.toString()}</p>
            );
        } else if (this.state.videoClient) {
            content = (
                <VideoStream
                    renderClient={this.state.renderClient}
                    videoClient={this.state.videoClient}
                />
            );
        }
        return <div className="App">{content}</div>;
    }
}

export default App;
