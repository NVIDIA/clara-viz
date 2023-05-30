# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.12)

include(CMakeParseArguments)

# Add custom commands to process ``.proto`` files to C++::
#
#     protobuf_grpc_generate ([TARGET <TGT>] [APPEND_PATH]
#         [LANGUAGE <LANG>] [OUT_VAR <VAR>] [EXPORT_MACRO <MACRO>]
#         [IMPORT_DIRS [dirs...]] [GENERATE_EXTENSIONS [exts...]
#         [PROTOS [files...]])
#
#   ``TARGET``
#     if specified the generated source files are added to the target
#   ``OUT_VAR``
#     if specified the generated source files are stored to a variable
#   ``APPEND_PATH``
#     create an include path for each file specified
#   ``LANGUAGE``
#     generated language, either 'python', 'grpc-web' or 'cpp', default 'cpp'
#   ``EXPORT_MACRO``
#     is a macro which should expand to ``__declspec(dllexport)`` or
#     ``__declspec(dllimport)`` depending on what is being compiled.
#   ``IMPORT_DIRS``
#     import directories
#   ``GENERATE_EXTENSIONS``
#     extensions of generated files, needed if an unrecognized language is specified
#   ``PROTOC_ARGS``
#     protoc argument, needed if an unrecognized language is specified
#   ``PROTOS``
#     ``.proto`` files
function(protobuf_grpc_generate)
  set(_options APPEND_PATH)
  set(_singleargs LANGUAGE OUT_VAR EXPORT_MACRO)
  if(COMMAND target_sources)
    list(APPEND _singleargs TARGET)
  endif()
  set(_multiargs PROTOS IMPORT_DIRS GENERATE_EXTENSIONS)

  cmake_parse_arguments(protobuf_generate "${_options}" "${_singleargs}" "${_multiargs}" "${ARGN}")

  if(NOT protobuf_generate_PROTOS AND NOT protobuf_generate_TARGET)
    message(SEND_ERROR "Error: protobuf_generate called without any targets or source files")
    return()
  endif()

  if(NOT protobuf_generate_OUT_VAR AND NOT protobuf_generate_TARGET)
    message(SEND_ERROR "Error: protobuf_generate called without a target or output variable")
    return()
  endif()

  if(NOT protobuf_generate_LANGUAGE)
    set(protobuf_generate_LANGUAGE cpp)
  endif()
  string(TOLOWER ${protobuf_generate_LANGUAGE} protobuf_generate_LANGUAGE)

  if(protobuf_generate_EXPORT_MACRO AND protobuf_generate_LANGUAGE STREQUAL cpp)
    set(_dll_export_decl "dllexport_decl=${protobuf_generate_EXPORT_MACRO}:")
  endif()

  if(protobuf_generate_LANGUAGE STREQUAL cpp)
    set(protobuf_PROTOC_ARGS --grpc_out ${_dll_export_decl}${CMAKE_CURRENT_BINARY_DIR} --cpp_out ${_dll_export_decl}${CMAKE_CURRENT_BINARY_DIR} --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>)
    set(protobuf_GENERATE_EXTENSIONS .pb.h .pb.cc .grpc.pb.h .grpc.pb.cc)
  elseif(protobuf_generate_LANGUAGE STREQUAL python)
    set(protobuf_PROTOC_ARGS --grpc_out ${_dll_export_decl}${CMAKE_CURRENT_BINARY_DIR} --python_out ${_dll_export_decl}${CMAKE_CURRENT_BINARY_DIR} --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_python_plugin>)
    set(protobuf_GENERATE_EXTENSIONS _pb2.py)
  elseif(protobuf_generate_LANGUAGE STREQUAL grpc-web)
    set(protobuf_PROTOC_ARGS --js_out=import_style=commonjs:${CMAKE_CURRENT_BINARY_DIR} --grpc-web_out=import_style=commonjs,mode=grpcwebtext:${CMAKE_CURRENT_BINARY_DIR} --plugin=protoc-gen-grpc-web=$<TARGET_FILE:grpc-web::grpc_grpc-web_plugin>)
    # Workaround for https://github.com/protocolbuffers/protobuf-javascript/issues/127
    # set path to protoc-gen-js
    set(protobuf_PROTOC_ARGS ${protobuf_PROTOC_ARGS} --plugin=protoc-gen-js=${protobuf-javascript_SOURCE_DIR}/bin/protoc-gen-js)
    set(protobuf_GENERATE_EXTENSIONS .js)
  else()
    if(NOT protobuf_GENERATE_EXTENSIONS OR NOT protobuf_PROTOC_ARGS)
      message(SEND_ERROR "Error: protobuf_generate given unknown Language ${LANGUAGE}, please provide a value for GENERATE_EXTENSIONS and PROTOC_ARGS")
      return()
    endif()
  endif()

  if(protobuf_generate_TARGET)
    get_target_property(_source_list ${protobuf_generate_TARGET} SOURCES)
    foreach(_file ${_source_list})
      if(_file MATCHES "proto$")
        list(APPEND protobuf_generate_PROTOS ${_file})
      endif()
    endforeach()
  endif()

  if(NOT protobuf_generate_PROTOS)
    message(SEND_ERROR "Error: protobuf_generate could not find any .proto files")
    return()
  endif()

  if(protobuf_generate_APPEND_PATH)
    # Create an include path for each file specified
    foreach(_file ${protobuf_generate_PROTOS})
      get_filename_component(_abs_file ${_file} ABSOLUTE)
      get_filename_component(_abs_path ${_abs_file} PATH)
      list(FIND _protobuf_include_path ${_abs_path} _contains_already)
      if(${_contains_already} EQUAL -1)
          list(APPEND _protobuf_include_path -I ${_abs_path})
      endif()
    endforeach()
  else()
    set(_protobuf_include_path -I ${CMAKE_CURRENT_SOURCE_DIR})
  endif()

  foreach(DIR IN LISTS protobuf_generate_IMPORT_DIRS)
    get_filename_component(ABS_PATH ${DIR} ABSOLUTE)
    list(FIND _protobuf_include_path ${ABS_PATH} _contains_already)
    if(${_contains_already} EQUAL -1)
        list(APPEND _protobuf_include_path -I ${ABS_PATH})
    endif()
  endforeach()

  set(_generated_srcs_all)
  foreach(_proto ${protobuf_generate_PROTOS})
    get_filename_component(_abs_file ${_proto} ABSOLUTE)
    get_filename_component(_directory ${_proto} DIRECTORY)
    get_filename_component(_basename ${_proto} NAME_WE)

    set(_generated_srcs)
    foreach(_ext ${protobuf_GENERATE_EXTENSIONS})
      list(APPEND _generated_srcs "${CMAKE_CURRENT_BINARY_DIR}/${_directory}/${_basename}${_ext}")
    endforeach()
    list(APPEND _generated_srcs_all ${_generated_srcs})

    add_custom_command(
      OUTPUT ${_generated_srcs}
      COMMAND  protobuf::protoc
      ARGS ${protobuf_PROTOC_ARGS}
        ${_protobuf_include_path}
        ${_abs_file}
      DEPENDS ${_abs_file} protobuf::protoc
      COMMENT "Running ${protobuf_generate_LANGUAGE} protocol buffer compiler on ${_proto}"
      VERBATIM )
  endforeach()

  set_source_files_properties(${_generated_srcs_all} PROPERTIES GENERATED TRUE)
  if(protobuf_generate_OUT_VAR)
    set(${protobuf_generate_OUT_VAR} ${_generated_srcs_all} PARENT_SCOPE)
  endif()
  if(protobuf_generate_TARGET)
    target_sources(${protobuf_generate_TARGET} PRIVATE ${_generated_srcs_all})
  endif()

endfunction()

# Add gRPC protobuf files. Generate sources and add dependencies
function(grpc_generate PROJECT_NAME)
    # protobuf sources are generate in the current binary directory, add
    # that directory to the include directories so we can include header
    # files
    set(CMAKE_INCLUDE_CURRENT_DIR TRUE PARENT_SCOPE)

    target_link_libraries(${PROJECT_NAME}
        PUBLIC
            gRPC::grpc++
        )

    target_include_directories(${PROJECT_NAME}
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        )

    # include the path to the proto files
    get_target_property(_include_paths ${PROJECT_NAME} INTERFACE_INCLUDE_DIRECTORIES)
    protobuf_grpc_generate(TARGET ${PROJECT_NAME} IMPORT_DIRS ${_include_paths})
endfunction()

# Prebuild setup - get the desired version
function(prebuild_setup)
    set(_options )
    set(_singleargs VERSION_VAR)
    set(_multiargs )

    cmake_parse_arguments(prebuild_setup "${_options}" "${_singleargs}" "${_multiargs}" "${ARGN}")

    if (NOT prebuild_setup_VERSION_VAR)
        message(SEND_ERROR "Version variable not specified")
        return()
    endif()

    # read the version
    file(READ ${CMAKE_CURRENT_SOURCE_DIR}/version.txt _version)
    set_directory_properties(PROPERTIES CMAKE_CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/version.txt)
    set(${prebuild_setup_VERSION_VAR} ${_version} PARENT_SCOPE)
endfunction()

# Builds the package name
function(prebuild_package_name)
    set(_options )
    set(_singleargs PREFIX VERSION VAR)
    set(_multiargs )

    cmake_parse_arguments(prebuild_package_name "${_options}" "${_singleargs}" "${_multiargs}" "${ARGN}")

    if (NOT prebuild_package_name_PREFIX)
        message(SEND_ERROR "Prefix not specified")
        return()
    endif()
    if (NOT prebuild_package_name_VERSION)
        message(SEND_ERROR "Version not specified")
        return()
    endif()
    if (NOT prebuild_package_name_VAR)
        message(SEND_ERROR "Result variable not specified")
        return()
    endif()
    if (DEFINED ENV{AUDITWHEEL_POLICY})
        set(_auditwheel_policy "-$ENV{AUDITWHEEL_POLICY}")
    else()
        set(_auditwheel_policy "")
    endif()

    # name of the package
    set(_package_name ${prebuild_package_name_PREFIX}-${prebuild_package_name_VERSION}-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}${_auditwheel_policy})
    set(${prebuild_package_name_VAR} ${_package_name} PARENT_SCOPE)
endfunction()

# Enable code coverage for the given project when the build target is 'Coverage'
function(target_code_coverage PROJECT_NAME)
    if (CMAKE_BUILD_TYPE STREQUAL "Coverage")
        target_compile_options(${PROJECT_NAME}
            PRIVATE
                # disable optimizations
                -O0
                # when compiling for Cuda keep the temporary files and pass the coverage option to gcc
                $<$<COMPILE_LANGUAGE:CUDA>:--keep --keep-dir ${CMAKE_CURRENT_BINARY_DIR} -Xcompiler>
                # compile and link code instrumented for coverage analysis
                --coverage
            )
        target_link_libraries(${PROJECT_NAME}
            PUBLIC
                gcov
            )
    endif()
endfunction()
