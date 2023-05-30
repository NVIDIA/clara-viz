/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
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
#include <iostream>
#include <string>

#include <getopt.h>

#include "ExampleRenderServer.h"

#include "claraviz/util/Exception.h"
#include "claraviz/util/Log.h"

using namespace clara::viz;

int main(int argc, char **argv)
{
    try
    {
        constexpr uint32_t DEFAULT_PORT                = 2050;
        constexpr uint32_t DEFAULT_CUDA_DEVICE_ORDINAL = 0;

        uint32_t port                = DEFAULT_PORT;
        uint32_t cuda_device_ordinal = DEFAULT_CUDA_DEVICE_ORDINAL;
        std::string mhd_file_name;

        struct option long_options[] = {{"help", no_argument, 0, 'h'},           {"file", required_argument, 0, 'f'},
                                        {"port", required_argument, 0, 'p'},     {"device", required_argument, 0, 'd'},
                                        {"loglevel", required_argument, 0, 'l'}, {0, 0, 0, 0}};

        // parse options
        while (true)
        {
            int option_index = 0;

            const int c = getopt_long(argc, argv, "hf:p:d:l:", long_options, &option_index);

            if (c == -1)
            {
                break;
            }

            const std::string argument(optarg ? optarg : "");
            switch (c)
            {
            case 'h':
                std::cout
                    << "RenderServer example. A basic volume renderer, the renderer output is streamed to "
                    << "a web client. Loading of MHD/MHA files (https://itk.org/Wiki/ITK/MetaIO/Documentation) is "
                    << "supported."
                    << "Usage: " << argv[0] << " [options]" << std::endl
                    << "Options:" << std::endl
                    << "  -h, --help                            Display this information" << std::endl
                    << "  -f <FILENAME>, --file <FILENAME>      Set the name of the MHD/MHA file for load" << std::endl
                    << "  -p <PORT>, --port <PORT>              Set the gRPC port to <PORT> (default " << DEFAULT_PORT
                    << ")" << std::endl
                    << "  -d <DEVICE>, --device <DEVICE>        Set the Cuda device to render on to <DEVICE> "
                       "(default "
                    << DEFAULT_CUDA_DEVICE_ORDINAL << ")" << std::endl
                    << "  -l <LOGLEVEL>, --loglevel <LOGLEVEL>  Set the loglevel to <LOGLEVEL>, available levels "
                       "'debug', 'info', 'warning' and 'error'; (default 'info')"
                    << std::endl;
                return EXIT_SUCCESS;

            case 'f':
                mhd_file_name = argument;
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
                cuda_device_ordinal = value;
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
                    Log::g_log_level = LogLevel::Debug;
                }
                else if (std::string(lower_case_argument) == "info")
                {
                    Log::g_log_level = LogLevel::Info;
                }
                else if (std::string(lower_case_argument) == "warning")
                {
                    Log::g_log_level = LogLevel::Warning;
                }
                else if (std::string(lower_case_argument) == "error")
                {
                    Log::g_log_level = LogLevel::Error;
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

        if (mhd_file_name.empty())
        {
            throw InvalidState() << "The name of the MHD/MHA file to load is required";
        }

        ExampleRenderServer renderServer(port, cuda_device_ordinal, mhd_file_name);

        renderServer.Run();

        renderServer.Wait();
    }
    catch (std::exception &er)
    {
        Log(LogLevel::Error) << "Error: " << er.what();
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
