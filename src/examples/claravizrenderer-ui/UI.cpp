/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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

#include "UI.h"

#include <cstring>

#include <X11/Xlib.h>
#include <X11/Xutil.h>

#include <stdexcept>

static bool exitApp = false;
static Atom WM_DELETE_WINDOW;

void processEvent(Display *display, Window window, const std::list<XImage *> &xColorImages,
                  const std::list<XImage *> &xAlphaImages, const std::list<XImage *> &xDepthImages, int width,
                  int height)
{
    XEvent ev;
    XNextEvent(display, &ev);
    switch (ev.type)
    {
    case Expose: {
        int dest_y = 0;
        for (auto &&xColorImage : xColorImages)
        {
            XPutImage(display, window, DefaultGC(display, 0), xColorImage, 0, 0, 0, dest_y, width, height);
            dest_y += height;
        }
        dest_y = 0;
        for (auto &&xAlphaImage : xAlphaImages)
        {
            XPutImage(display, window, DefaultGC(display, 0), xAlphaImage, 0, 0, width, dest_y, width, height);
            dest_y += height;
        }
        dest_y = 0;
        for (auto &&xDepthImage : xDepthImages)
        {
            XPutImage(display, window, DefaultGC(display, 0), xDepthImage, 0, 0, width * 2, dest_y, width, height);
            dest_y += height;
        }
    }
    break;
    case ButtonPress:
        exitApp = true;
        break;
    case ClientMessage:
        if ((Atom)ev.xclient.data.l[0] == WM_DELETE_WINDOW)
        {
            exitApp = true;
        }
        break;
    }
}

XImage *createColorImage(Display *display, Visual *visual, int width, int height, const std::vector<uint8_t> &data)
{
    char *image32 = reinterpret_cast<char *>(malloc(data.size()));
    std::memcpy(image32, data.data(), width * height * 4);
    return XCreateImage(display, visual, DefaultDepth(display, DefaultScreen(display)), ZPixmap, 0, image32, width,
                        height, 32, 0);
}

XImage *createAlphaImage(Display *display, Visual *visual, int width, int height, const std::vector<uint8_t> &data)
{
    char *image32 = reinterpret_cast<char *>(malloc(data.size()));
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const uint8_t source             = data[(x + y * width) * 4 + 3];
            image32[(x + y * width) * 4 + 0] = source;
            image32[(x + y * width) * 4 + 1] = source;
            image32[(x + y * width) * 4 + 2] = source;
            image32[(x + y * width) * 4 + 3] = 255;
        }
    }
    return XCreateImage(display, visual, DefaultDepth(display, DefaultScreen(display)), ZPixmap, 0, image32, width,
                        height, 32, 0);
}

XImage *createDepthImage(Display *display, Visual *visual, int width, int height, const std::vector<float> &data)
{
    char *image32 = reinterpret_cast<char *>(malloc(width * height * 4));
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const uint8_t source = static_cast<uint8_t>((std::min(data[x + y * width], 1.f) * 255.f) + 0.5f);
            image32[(x + y * width) * 4 + 0] = source;
            image32[(x + y * width) * 4 + 1] = source;
            image32[(x + y * width) * 4 + 2] = source;
            image32[(x + y * width) * 4 + 3] = 255;
        }
    }
    return XCreateImage(display, visual, DefaultDepth(display, DefaultScreen(display)), ZPixmap, 0, image32, width,
                        height, 32, 0);
}

void show(unsigned int width, unsigned int height, const std::list<std::vector<uint8_t>> &data_list,
          const std::list<std::vector<float>> &depth_data_list)
{
    // open the window
    Display *display = XOpenDisplay(NULL);
    if (!display)
    {
        throw std::runtime_error("XOpenDisplay failed");
    }
    Visual *visual = DefaultVisual(display, 0);
    if (!visual)
    {
        throw std::runtime_error("DefaultVisual failed");
    }
    Window window =
        XCreateSimpleWindow(display, RootWindow(display, 0), 0, 0, width * 3, height * data_list.size(), 1, 0, 0);
    if (!window)
    {
        throw std::runtime_error("XCreateSimpleWindow failed");
    }

    if (!XSelectInput(display, window, ButtonPressMask | ExposureMask))
    {
        throw std::runtime_error("XSelectInput failed");
    }
    if (!XMapWindow(display, window))
    {
        throw std::runtime_error("XMapWindow failed");
    }

    WM_DELETE_WINDOW = XInternAtom(display, "WM_DELETE_WINDOW", False);
    XSetWMProtocols(display, window, &WM_DELETE_WINDOW, 1);

    std::list<XImage *> xColorImages;
    std::list<XImage *> xAlphaImages;
    std::list<XImage *> xDepthImages;

    for (auto &&data : data_list)
    {
        XImage *xColorImage = createColorImage(display, visual, width, height, data);
        if (!xColorImage)
        {
            throw std::runtime_error("XCreateImage failed");
        }
        xColorImages.push_back(xColorImage);

        XImage *xAlphaImage = createAlphaImage(display, visual, width, height, data);
        if (!xAlphaImage)
        {
            throw std::runtime_error("XCreateImage failed");
        }
        xAlphaImages.push_back(xAlphaImage);
    }

    for (auto &&data : depth_data_list)
    {
        XImage *xDepthImage = createDepthImage(display, visual, width, height, data);
        if (!xDepthImage)
        {
            throw std::runtime_error("XCreateImage failed");
        }
        xDepthImages.push_back(xDepthImage);
    }

    while (!exitApp)
    {
        processEvent(display, window, xColorImages, xAlphaImages, xDepthImages, width, height);
    }

    for (auto &&xDepthImage : xDepthImages)
    {
        if (!XDestroyImage(xDepthImage))
        {
            throw std::runtime_error("XDestroyImage failed");
        }
    }

    for (auto &&xAlphaImage : xAlphaImages)
    {
        if (!XDestroyImage(xAlphaImage))
        {
            throw std::runtime_error("XDestroyImage failed");
        }
    }

    for (auto &&xColorImage : xColorImages)
    {
        if (!XDestroyImage(xColorImage))
        {
            throw std::runtime_error("XDestroyImage failed");
        }
    }

    if (!XDestroyWindow(display, window))
    {
        throw std::runtime_error("XDestroyWindow failed");
    }

    XCloseDisplay(display);
}
