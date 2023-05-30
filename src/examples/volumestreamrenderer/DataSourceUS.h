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

#pragma once

#include "DataSource.h"

namespace clara::viz
{

class DataSourceUS : public DataSource
{
public:
    /**
     * Construct
     * 
     * @param input_dir [in] source data input directory, if empty generate synthetic data
     */
    DataSourceUS(const std::string input_dir);

    float FrameRate() override;

    std::shared_ptr<IBlob> NextBlob(bool benchmark_mode) override;
    void ReturnBlob(const std::shared_ptr<IBlob> &blob) override;

    void Light(LightInterface &interface) override;
    void BackgroundLight(BackgroundLightInterface &interface) override;
    void Camera(CameraInterface &interface) override;
    void TransferFunction(TransferFunctionInterface &interface) override;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace clara::viz
