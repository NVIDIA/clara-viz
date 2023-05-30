/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
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

#include "claraviz/core/Image.h"

#include <vector>

#include "claraviz/hardware/cuda/CudaService.h"
#include "claraviz/interface/ImageInterface.h"
#include "claraviz/image/JpegEncoder.h"
#include "claraviz/util/Thread.h"
#include "claraviz/util/StdContainerBlob.h"

namespace clara::viz
{

DEFINE_CLASS_MESSAGEID(ImageMessage);
DEFINE_CLASS_MESSAGEID(ImageEncodeMessage);
DEFINE_CLASS_MESSAGEID(ImageRenderFailedMessage);
DEFINE_CLASS_MESSAGEID(ImageEncodedDataMessage);

namespace
{

/**
 * Shutdown message ID
 */
DEFINE_MESSAGEID(IDMESSAGE_SHUTDOWN);

} // anonymous namespace

/**
 * Server implementation data
 */
class Image::Impl
{
public:
    explicit Impl(const std::shared_ptr<Image> &base, const std::shared_ptr<MessageReceiver> &image_interface_output,
                  uint32_t cuda_device_ordinal);
    Impl() = delete;

    void Run();
    void Shutdown();

private:
    /**
     * Interface thread function.
     */
    void ThreadFunction(std::function<void()> ready);

    /**
     * Send to encoder.
     *
     * @param message [in]
     */
    void Encode(std::shared_ptr<const ImageEncodeMessage> message);

    /// base class (needs to be a weak_ptr to break circular dependency)
    const std::weak_ptr<Image> base_;

    /// message receiver for the encoded data
    std::shared_ptr<MessageReceiver> image_interface_output_;

    /**
     * Cuda context.
     * Note that the context has to be the first member interacting with Cuda.
     * The context will be destroyed last and all other Cuda objects should be already
     * be destroyed at this point.
     */
    std::unique_ptr<CudaPrimaryContext> cuda_context_;

    /// The cuda device to render on
    const uint32_t cuda_device_ordinal_;

    /// interface thread
    std::unique_ptr<Thread> thread_;

    /// JPEG encoder
    std::unique_ptr<JpegEncoder> encoder_;
};

std::shared_ptr<Image> Image::Create(const std::shared_ptr<MessageReceiver> &image_interface_output,
                                     uint32_t cuda_device_ordinal)
{
    auto image = std::make_shared<Image>();

    image->impl_.reset(new Image::Impl(image, image_interface_output, cuda_device_ordinal));

    return image;
}

Image::~Image()
{
    Shutdown();
}

void Image::Run()
{
    impl_->Run();
}

void Image::Shutdown()
{
    if (impl_)
    {
        impl_->Shutdown();
    }
}

Image::Impl::Impl(const std::shared_ptr<Image> &base, const std::shared_ptr<MessageReceiver> &image_interface_output,
                  uint32_t cuda_device_ordinal)
    : base_(base)
    , image_interface_output_(image_interface_output)
    , cuda_device_ordinal_(cuda_device_ordinal)
{
    if (!base)
    {
        throw InvalidArgument("base") << "is nullptr";
    }
}

void Image::Impl::Run()
{
    // run the render thread
    thread_.reset(new Thread("Encoder thread", [this](std::function<void()> ready) { ThreadFunction(ready); }));
}

void Image::Impl::Shutdown()
{
    if (thread_)
    {
        // might be called when the base class is already destroyed
        const std::shared_ptr<Image> base = base_.lock();
        if (base)
        {
            base->EnqueueMessage(std::make_shared<Message>(IDMESSAGE_SHUTDOWN));
        }

        // destroy the thread
        thread_.reset();
    }
}

void Image::Impl::ThreadFunction(std::function<void()> ready)
{
    // initialize Cuda
    CudaCheck(cuInit(0));
    cuda_context_.reset(new CudaPrimaryContext(cuda_device_ordinal_));

    // create the encoder
    encoder_.reset(new JpegEncoder());

    // thread is ready now
    ready();

    while (true)
    {
        try
        {
            auto base = base_.lock();

            base->Wait();

            bool shutdown = false;
            std::shared_ptr<const Message> message;
            while ((message = base->DequeueMessage()))
            {
                Log(LogLevel::Debug) << "Image received " << message->GetID().GetName();

                if (message->GetID() == ImageInterface::Message::id_)
                {
                    // interface message
                    const ImageInterface::DataOut &image_data_ =
                        std::static_pointer_cast<const ImageInterface::Message>(message)->data_out_;

                    // ignore initial state message
                    if (image_data_.color_type == ColorImageType::UNKNOWN)
                    {
                        break;
                    }

                    // forward the data to the client
                    auto image_message = std::make_shared<ImageMessage>();

                    image_message->view_name_  = image_data_.view_name;
                    image_message->width_      = image_data_.width;
                    image_message->height_     = image_data_.height;
                    image_message->color_type_ = image_data_.color_type;
                    if (image_data_.color_type == ColorImageType::JPEG)
                    {
                        image_message->parameters_.jpeg_.quality_ = image_data_.jpeg_quality;
                    }
                    image_message->color_memory_ = image_data_.color_memory;
                    image_message->depth_type_   = image_data_.depth_type;
                    image_message->depth_memory_ = image_data_.depth_memory;

                    base->SendMessage(image_message);
                }
                else if (message->GetID() == ImageEncodeMessage::id_)
                {
                    // client message (from renderer), encode a frame
                    Encode(std::static_pointer_cast<const ImageEncodeMessage>(message));
                }
                else if (message->GetID() == ImageRenderFailedMessage::id_)
                {
                    // send a empty data message to the client
                    auto encoded_data_message         = std::make_shared<ImageEncodedDataMessage>();
                    encoded_data_message->color_type_ = ColorImageType::UNKNOWN;
                    encoded_data_message->depth_type_ = DepthImageType::UNKNOWN;
                    image_interface_output_->EnqueueMessage(encoded_data_message);
                }
                else if (message->GetID() == IDMESSAGE_SHUTDOWN)
                {
                    shutdown = true;
                }
                else
                {
                    throw InvalidState() << "Unhandled message Id " << message->GetID().GetName();
                }
            }

            if (shutdown)
            {
                break;
            }
        }
        catch (const std::exception &e)
        {
            Log(LogLevel::Error) << "Image thread threw exception " << e.what();
        }
        catch (...)
        {
            Log(LogLevel::Error) << "Image thread threw unknown exception";
        }
    }
}

void Image::Impl::Encode(std::shared_ptr<const ImageEncodeMessage> message)
{
    auto encoded_data_message = std::make_shared<ImageEncodedDataMessage>();

    auto base = base_.lock();

    // check if there is color data and we should encode to JPEG, do it
    if (message->color_memory_)
    {
        if (message->color_type_ == ColorImageType::JPEG)
        {
            // encode the image
            JpegEncoder::Format encoder_format;
            switch (message->color_format_)
            {
            case ImageEncodeMessage::ColorFormat::ABGR_U8:
                encoder_format = JpegEncoder::Format::ABGR;
                break;
            default:
                throw InvalidState() << "Unhandled image format " << static_cast<int>(message->color_format_);
            }

            encoder_->SetQuality(message->parameters_.jpeg_.quality_);
            std::unique_ptr<std::vector<uint8_t>> data(new std::vector<uint8_t>());
            encoder_->Encode(message->width_, message->height_, encoder_format, message->color_memory_, *data.get());
            encoded_data_message->color_memory_.reset(new StdContainerBlob<std::vector<uint8_t>>(std::move(data)));
        }
        else
        {
            // else just pass it along
            encoded_data_message->color_memory_ = message->color_memory_;
        }
    }
    encoded_data_message->color_type_ = message->color_type_;

    encoded_data_message->depth_memory_ = message->depth_memory_;
    encoded_data_message->depth_type_   = message->depth_type_;

    // send the data to the image data receiver
    image_interface_output_->EnqueueMessage(encoded_data_message);
}

} // namespace clara::viz
