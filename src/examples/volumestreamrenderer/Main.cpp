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

#include <algorithm>
#include <cctype>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include <getopt.h>

#include "VolumeStreamRenderServer.h"
#include "claraviz/util/Exception.h"
#include "claraviz/util/Log.h"

int main(int argc, char **argv)
{
    try
    {
        const std::string DEFAULT_SCENARIO = "CT";
        constexpr std::chrono::seconds DEFAULT_BENCHMARK_DURATION(5);
        constexpr bool DEFAULT_STREAM_FROM_CPU         = false;
        constexpr uint32_t DEFAULT_PORT                = 2050;
        constexpr uint32_t DEFAULT_CUDA_DEVICE_ORDINAL = 0;

        std::string input_dir;
        std::string scenario = DEFAULT_SCENARIO;
        std::chrono::seconds benchmark_duration(0);
        bool stream_from_cpu = DEFAULT_STREAM_FROM_CPU;
        uint32_t port        = DEFAULT_PORT;
        std::vector<uint32_t> cuda_device_ordinals;

        struct option long_options[] = {{"help", no_argument, 0, 'h'},
                                        {"input", required_argument, 0, 'i'},
                                        {"scenario", required_argument, 0, 's'},
                                        {"benchmark", optional_argument, 0, 'b'},
                                        {"streamcpu", no_argument, 0, 'c'},
                                        {"port", required_argument, 0, 'p'},
                                        {"device", required_argument, 0, 'd'},
                                        {"loglevel", required_argument, 0, 'l'},
                                        {0, 0, 0, 0}};

        // parse options
        while (true)
        {
            int option_index = 0;

            const int c = getopt_long(argc, argv, "hi:s:b::p:d:l:c", long_options, &option_index);

            if (c == -1)
            {
                break;
            }

            const std::string argument(optarg ? optarg : "");
            switch (c)
            {
            case 'h':
                std::cout
                    << "Usage: " << argv[0] << " [options]" << std::endl
                    << "Options:" << std::endl
                    << "  -h, --help                            Display this information" << std::endl
                    << "  -s <SCENARIO>, --scenario <SCENARIO>  Scenario, either 'CT' or 'US' (default "
                    << DEFAULT_SCENARIO << ")" << std::endl
                    << "  -i <DIRECTORY> --input <DIRECTORY>    Input directory for data files, see README.md for the "
                       "expected data file organization. If no input is specified then synthetic data is generated."
                    << std::endl
                    << "  -b[<TIME>], --benchmark[<TIME>]       Run in benchmark mode for <TIME> seconds (default "
                    << DEFAULT_BENCHMARK_DURATION.count() << ")" << std::endl
                    << "  -c, --streamcpu                       Stream from CPU (host) memory, else from GPU (device) "
                       "memory (for 'CT' scenario only)"
                    << std::endl
                    << "  -p <PORT>, --port <PORT>              Set the gRPC port to <PORT> (default " << DEFAULT_PORT
                    << ")" << std::endl
                    << "  -d <DEVICE>, --device <DEVICE>        Add <DEVIVICE> to the list of Cuda device to render "
                       "(default "
                    << DEFAULT_CUDA_DEVICE_ORDINAL << ")" << std::endl
                    << "  -l <LOGLEVEL>, --loglevel <LOGLEVEL>  Set the loglevel to <LOGLEVEL>, available levels "
                       "'debug', 'info', 'warning' and 'error'; (default 'info')"
                    << std::endl;
                return EXIT_SUCCESS;

            case 'i':
                input_dir = argument;
                break;

            case 's':
                if ((argument != "CT") && (argument != "US"))
                {
                    throw InvalidArgument("scenario") << "Invalid scenario " << argument;
                }
                scenario = argument;
                break;

            case 'b':
                if (optarg)
                {
                    const int value = std::stoi(argument);
                    if (value < 0)
                    {
                        throw InvalidArgument("benchmark")
                            << "Invalid benchmark duration " << value << ", must be greater than zero";
                    }
                    benchmark_duration = std::chrono::seconds(value);
                }
                else
                {
                    benchmark_duration = DEFAULT_BENCHMARK_DURATION;
                }
                break;

            case 'c':
                stream_from_cpu = true;
                break;

            case 'p':
            {
                const int value = std::stoi(argument);
                if ((value < 0) || (value > 65535))
                {
                    throw InvalidArgument("port") << "Invalid port " << value << ", must be in the range of [1, 65535]";
                }
                port = value;
                break;
            }

            case 'd':
            {
                const int value = std::stoi(argument);
                if (value < 0)
                {
                    throw InvalidArgument("device") << "Invalid device " << value << ", must be >= 0";
                }
                cuda_device_ordinals.push_back(value);
                break;
            }

            case 'l':
            {
                // convert to lower case
                std::string lower_case_argument;
                std::transform(argument.begin(), argument.end(), std::back_inserter(lower_case_argument),
                               [](unsigned char c) -> unsigned char { return std::tolower(c); });
                if (std::string(lower_case_argument) == "debug")
                {
                    clara::viz::Log::g_log_level = clara::viz::LogLevel::Debug;
                }
                else if (std::string(lower_case_argument) == "info")
                {
                    clara::viz::Log::g_log_level = clara::viz::LogLevel::Info;
                }
                else if (std::string(lower_case_argument) == "warning")
                {
                    clara::viz::Log::g_log_level = clara::viz::LogLevel::Warning;
                }
                else if (std::string(lower_case_argument) == "error")
                {
                    clara::viz::Log::g_log_level = clara::viz::LogLevel::Error;
                }
                else
                {
                    throw InvalidArgument("loglevel") << "Invalid log level '" << argument << "'";
                }
                break;
            }
            case '?':
                // unknown option, error already printed by getop_long
                break;
            default:
                throw InvalidState() << "Unhandled option " << c;
            }
        }

        // if no device had been specified, add the default device
        if (cuda_device_ordinals.empty())
        {
            cuda_device_ordinals.push_back(DEFAULT_CUDA_DEVICE_ORDINAL);
        }

        clara::viz::VolumeStreamRenderServer renderServer(input_dir, scenario, benchmark_duration, stream_from_cpu, port,
                                                    cuda_device_ordinals);

        renderServer.Run();

        renderServer.Wait();
    }
    catch (std::exception &er)
    {
        clara::viz::Log(clara::viz::LogLevel::Error) << "Error: " << er.what();
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
