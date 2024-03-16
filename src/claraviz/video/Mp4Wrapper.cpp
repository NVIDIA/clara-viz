/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "claraviz/video/Mp4Wrapper.h"

#include <string>
#include <algorithm>

#include "claraviz/util/Exception.h"

/**
 * Documentation:
 *
 * Fragmented MPEG-4:
 *  https://tools.ietf.org/html/draft-pantos-http-live-streaming-23#section-3.3
 * MP4 layout
 *  http://xhelmboyx.tripod.com/formats/mp4-layout.txt
 */

namespace clara::viz
{

using Mp4AtomHandle = std::shared_ptr<class Mp4Atom>;

struct Nalu
{
    std::vector<uint8_t>::const_iterator nalubegin_;
    std::vector<uint8_t>::const_iterator naluend_;

    inline uint8_t nalutypeh264() const
    {
        const uint8_t header  = *nalubegin_;
        uint8_t nal_ref_idc   = header & 0x60;
        uint8_t nal_unit_type = header & 0x1f;
        return nal_unit_type;
    }

    static const uint8_t H264_SPS_TYPE = 7;
    static const uint8_t H264_PPS_TYPE = 8;

    inline uint8_t nalutypehevc() const
    {
        const uint8_t header  = *nalubegin_;
        uint8_t nal_unit_type = (header >> 1) & 0x3f;
        return nal_unit_type;
    }

    static const uint8_t HEVC_SPS_TYPE = 33;
    static const uint8_t HEVC_PPS_TYPE = 34;
};

/// sample flags
struct SampleFlags
{
    SampleFlags()
        : all(0)
    {
    }

    union
    {
        struct
        {
            uint32_t sample_degradation_priority : 16;
            uint32_t sample_is_non_sync_sample : 1;
            uint32_t sample_padding_value : 3;
            uint32_t sample_has_redundancy : 2;
            uint32_t sample_is_depended_on : 2;
            uint32_t sample_depends_on : 2;
            uint32_t is_leading : 2;
            uint32_t reserved : 4;
        };
        uint32_t all;
    };
};

/// the leading nature of this sample is unknown
constexpr uint32_t SAMPLE_FLAGS_IS_LEADING_UNKNOWN = 0;
/// this sample is a leading sample that has a dependency before the referenced I-picture (and is therefore not decodable)
constexpr uint32_t SAMPLE_FLAGS_IS_LEADING_DEPENDENCY_BEFORE = 1;
/// this sample is not a leading sample
constexpr uint32_t SAMPLE_FLAGS_IS_LEADING_NOT = 2;
/// this sample is a leading sample that has no dependency before the referenced I-picture (and is therefore decodable)
constexpr uint32_t SAMPLE_FLAGS_IS_LEADING_NO_DEPENDENCY_BEFORE = 3;

// the dependency of this sample is unknown
constexpr uint32_t SAMPLE_FLAGS_DEPENDS_UNKNOWN = 0;
// this sample does depend on others (not an I picture)
constexpr uint32_t SAMPLE_FLAGS_DEPENDS_ON_OTHERS = 1;
// this sample does not depend on others (I picture)
constexpr uint32_t SAMPLE_FLAGS_DEPENDS_ON_NO_OTHERS = 2;

// the dependency of other samples on this sample is unknown
constexpr uint32_t SAMPLE_FLAGS_IS_DEPENDEND_UNKNOWN = 0;
// other samples may depend on this one (not disposable)
constexpr uint32_t SAMPLE_FLAGS_IS_DEPENDEND_OTHER = 1;
// no other sample depends on this one (disposable)
constexpr uint32_t SAMPLE_FLAGS_IS_DEPENDEND_NO_OTHER = 2;

// it is unknown whether there is redundant coding in this sample
constexpr uint32_t SAMPLE_FLAGS_HAS_REDUNDANCY_UNKNOWN = 0;
// there is redundant coding in this sample
constexpr uint32_t SAMPLE_FLAGS_HAS_REDUNDANCY_CODING = 1;
// there is no redundant coding in this sample
constexpr uint32_t SAMPLE_FLAGS_HAS_REDUNDANCY_NOT_CODING = 2;

class Mp4Atom
{
public:
    static inline uint32_t CreateAtomTag(const char *tag)
    {
        return (static_cast<uint32_t>(tag[0]) << 24) | (static_cast<uint32_t>(tag[1]) << 16) |
               (static_cast<uint32_t>(tag[2]) << 8) | static_cast<uint32_t>(tag[3]);
    }

    inline uint32_t GetHeaderSize() const
    {
        return sizeof(header_);
    }

    virtual uint32_t GetBodySize() const
    {
        return 0;
    }

    Mp4Atom(const char *type)
        : header_(type)
    {
    }
    virtual ~Mp4Atom() {}

    virtual void WriteAtom(std::vector<uint8_t> &buffer)
    {
        WriteAtomHeader(buffer);
        WriteAtomChildren(buffer);
    };

    uint32_t GetSize() const
    {
        return header_.size_;
    }

    virtual void UpdateSize()
    {
        uint32_t size = GetHeaderSize() + GetBodySize();
        for (size_t idx = 0; idx < children_.size(); idx++)
        {
            size += children_[idx]->GetSize();
        }

        SetSize(size);
    }

    inline void SetSize(uint32_t size)
    {
        header_.size_ = size;
        if (parent_)
        {
            parent_->UpdateSize();
        }
    }

    inline void WriteValue(std::vector<uint8_t> &buffer, uint64_t value)
    {
        uint8_t *data = reinterpret_cast<uint8_t *>(&value);
        buffer.push_back(data[7]);
        buffer.push_back(data[6]);
        buffer.push_back(data[5]);
        buffer.push_back(data[4]);
        buffer.push_back(data[3]);
        buffer.push_back(data[2]);
        buffer.push_back(data[1]);
        buffer.push_back(data[0]);
    }

    inline void WriteValue(std::vector<uint8_t> &buffer, uint32_t value)
    {
        uint8_t *data = reinterpret_cast<uint8_t *>(&value);
        buffer.push_back(data[3]);
        buffer.push_back(data[2]);
        buffer.push_back(data[1]);
        buffer.push_back(data[0]);
    }

    inline void WriteValue(std::vector<uint8_t> &buffer, uint16_t value)
    {
        uint8_t *data = reinterpret_cast<uint8_t *>(&value);
        buffer.push_back(data[1]);
        buffer.push_back(data[0]);
    }

    inline void WriteValue(std::vector<uint8_t> &buffer, uint8_t value)
    {
        buffer.push_back(value);
    }

    inline void WriteValue64(std::vector<uint8_t> &buffer, uint64_t value)
    {
        WriteValue(buffer, value);
    }
    inline void WriteValue32(std::vector<uint8_t> &buffer, uint32_t value)
    {
        WriteValue(buffer, value);
    }
    inline void WriteValue16(std::vector<uint8_t> &buffer, uint16_t value)
    {
        WriteValue(buffer, value);
    }
    inline void WriteValue8(std::vector<uint8_t> &buffer, uint8_t value)
    {
        WriteValue(buffer, value);
    }

    inline void WriteAtomHeader(std::vector<uint8_t> &buffer)
    {
        WriteValue32(buffer, header_.size_);
        WriteValue32(buffer, header_.type_);
    }

    inline void WriteAtomChildren(std::vector<uint8_t> &buffer)
    {
        for(auto &child: children_)
        {
            child->WriteAtom(buffer);
        }
    }

    inline uint32_t GetStringSize(const std::string &str) const
    {
        if (str.size() == 0)
        {
            return 0;
        }
        else
        {
            return static_cast<uint32_t>(str.size() + 1);
        }
    }

    inline void WriteString(std::vector<uint8_t> &buffer, const std::string &str)
    {
        buffer.insert(buffer.end(), str.begin(), str.end());
        size_t size = static_cast<size_t>(GetStringSize(str));
        buffer.insert(buffer.end(), size - str.size(), 0);
    }

    inline void WriteString(std::vector<uint8_t> &buffer, const std::string &str, uint32_t size)
    {
        size_t idx = 0;
        for (; idx < str.size() && idx < size; ++idx)
        {
            buffer.push_back((uint8_t)str[idx]);
        }
        for (; idx < size; ++idx)
        {
            buffer.push_back(0);
        }
    }

    inline void WriteData(std::vector<uint8_t> &buffer, std::vector<uint8_t>::const_iterator begin,
                          std::vector<uint8_t>::const_iterator end)
    {
        buffer.insert(buffer.end(), begin, end);
    }

    inline void AddChild(Mp4AtomHandle child)
    {
        children_.push_back(child);
        child->parent_ = this;
        UpdateSize();
    }

protected:
    struct __attribute__((__packed__)) Header
    {
        explicit Header(const char *type)
            : type_(CreateAtomTag(type))
            , size_(sizeof(*this))
        {
        }
        Header() = delete;

        const uint32_t type_;
        uint32_t size_;
    } header_;

    Mp4Atom *parent_ = nullptr;
    std::vector<Mp4AtomHandle> children_;
};

class Mp4VersionedAtom : public Mp4Atom
{
public:
    Mp4VersionedAtom(const char *type, uint8_t version, uint32_t flags)
        : Mp4Atom(type)
        , flags_((static_cast<uint32_t>(version) << 24) | (flags & 0xffffff))
    {
    }

    void WriteAtomHeader(std::vector<uint8_t> &buffer)
    {
        WriteValue32(buffer, header_.size_);
        WriteValue32(buffer, header_.type_);
        WriteValue32(buffer, flags_);
    }

    inline uint32_t GetHeaderSize() const
    {
        return Mp4Atom::GetHeaderSize() + sizeof(flags_);
    }

    virtual void UpdateSize() override
    {
        uint32_t size = GetHeaderSize() + GetBodySize();
        for (size_t idx = 0; idx < children_.size(); idx++)
        {
            size += children_[idx]->GetSize();
        }

        SetSize(size);
    }

protected:
    uint32_t flags_;
};

/// Media Data Container
class MDAT_Atom : public Mp4Atom
{
public:
    MDAT_Atom()
        : Mp4Atom("mdat")
        , nalus_(nullptr)
    {
    }

    virtual void UpdateSize() override
    {
        // SetNalus will set the correct size
    }

    void SetNalus(std::vector<Nalu> *nalus, size_t size)
    {
        nalus_ = nalus;
        SetSize(static_cast<uint32_t>(size) + GetHeaderSize());
    }

    virtual void WriteAtom(std::vector<uint8_t> &buffer) override
    {
        WriteAtomHeader(buffer);

        for (auto nalu : (*nalus_))
        {
            uint32_t nalusize4 = static_cast<uint32_t>(nalu.naluend_ - nalu.nalubegin_);
            WriteValue(buffer, nalusize4);

            //copy entire NALU in one go, without skipping emulation prevention bytes
            WriteData(buffer, nalu.nalubegin_, nalu.naluend_);
        }
    }

private:
    std::vector<Nalu> *nalus_;
};

/// Track Fragment Run Box
class TRUN_Atom : public Mp4VersionedAtom
{
public:
    static const uint32_t DATA_OFFSET_C                     = 0x1;
    static const uint32_t FIRST_SAMPLE_FLAGS_C              = 0x4;
    static const uint32_t SAMPLE_DURATION_C                 = 0x100;
    static const uint32_t SAMPLE_SIZE_C                     = 0x200;
    static const uint32_t SAMPLE_FLAGS_C                    = 0x400;
    static const uint32_t SAMPLE_COMPOSITION_TIME_OFFSETS_C = 0x800;

    TRUN_Atom(uint32_t data_offset, uint32_t sample_size, SampleFlags first_sample_flags)
        : Mp4VersionedAtom("trun", 0, DATA_OFFSET_C | FIRST_SAMPLE_FLAGS_C | SAMPLE_SIZE_C)
        , body_(data_offset, sample_size, first_sample_flags)
    {
        UpdateSize();
    }

    virtual uint32_t GetBodySize() const override
    {
        return sizeof(Body);
    }

    virtual void WriteAtom(std::vector<uint8_t> &buffer) override
    {
        WriteAtomHeader(buffer);
        WriteValue(buffer, body_.sample_count_);
        WriteValue(buffer, body_.data_offset_);
        WriteValue(buffer, body_.first_sample_flags_);
        WriteValue(buffer, body_.sample_size_);
    }

    void SetDataOffset(uint32_t data_offset)
    {
        body_.data_offset_ = data_offset;
    }

    void SetSampleSize(uint32_t sample_size)
    {
        body_.sample_size_ = sample_size;
    }

private:
    struct __attribute__((__packed__)) Body
    {
        explicit Body(uint32_t data_offset, uint32_t sample_size, SampleFlags first_sample_flags)
            : data_offset_(data_offset)
            , first_sample_flags_(first_sample_flags.all)
            , sample_size_(sample_size)
        {
        }
        Body() = delete;

        const uint32_t sample_count_ = 1;
        uint32_t data_offset_;
        const uint32_t first_sample_flags_;
        uint32_t sample_size_;
    } body_;
};

/// Track fragment decode time
class TFDT_Atom : public Mp4VersionedAtom
{
public:
    TFDT_Atom(uint64_t base_media_decode_time)
        : Mp4VersionedAtom("tfdt", 1, 0)
        , body_(base_media_decode_time)
    {
        UpdateSize();
    }

    virtual uint32_t GetBodySize() const override
    {
        return sizeof(Body);
    }

    virtual void WriteAtom(std::vector<uint8_t> &buffer) override
    {
        WriteAtomHeader(buffer);
        WriteValue(buffer, body_.base_media_decode_time_);
    }

    void SetBaseMediaDecodeTime(uint64_t base_media_decode_time)
    {
        body_.base_media_decode_time_ = base_media_decode_time;
    }

private:
    struct __attribute__((__packed__)) Body
    {
        explicit Body(uint64_t base_media_decode_time)
            : base_media_decode_time_(base_media_decode_time)
        {
        }
        Body() = delete;

        uint64_t base_media_decode_time_;
    } body_;
};

/// Track Fragment Header Box
class TFHD_Atom : public Mp4VersionedAtom
{
public:
    static const uint32_t IS_MOOF_C                 = 0x20000;
    static const uint32_t DEFAULT_SAMPLE_DURATION_C = 0x00008;
    static const uint32_t DEFAULT_SAMPLE_SIZE_C     = 0x00010;
    static const uint32_t DEFAULT_SAMPLE_FLAGS_C    = 0x00020;

    TFHD_Atom(uint32_t track_id, uint32_t default_sample_duration, uint32_t default_sample_size,
              SampleFlags default_sample_flags)
        : Mp4VersionedAtom("tfhd", 0,
                           IS_MOOF_C | DEFAULT_SAMPLE_DURATION_C | DEFAULT_SAMPLE_SIZE_C | DEFAULT_SAMPLE_FLAGS_C)
        , body_(track_id, default_sample_duration, default_sample_size, default_sample_flags)
    {
        UpdateSize();
    }

    virtual uint32_t GetBodySize() const override
    {
        return sizeof(Body);
    }

    virtual void WriteAtom(std::vector<uint8_t> &buffer) override
    {
        WriteAtomHeader(buffer);
        WriteValue(buffer, body_.track_id_);
        WriteValue(buffer, body_.default_sample_duration_);
        WriteValue(buffer, body_.default_sample_size_);
        WriteValue(buffer, body_.default_sample_flags_);
    }

    void SetDefaultSampleSize(uint32_t default_sample_size)
    {
        body_.default_sample_size_ = default_sample_size;
    }

    /// Set the default sample flags
    /// \param default_sample_flags sample flags
    void SetDefaultSampleFlags(SampleFlags default_sample_flags)
    {
        body_.default_sample_flags_ = default_sample_flags.all;
    }

private:
    struct __attribute__((__packed__)) Body
    {
        explicit Body(uint32_t track_id, uint32_t default_sample_duration, uint32_t default_sample_size,
                      SampleFlags default_sample_flags)
            : track_id_(track_id)
            , default_sample_duration_(default_sample_duration)
            , default_sample_size_(default_sample_size)
            , default_sample_flags_(default_sample_flags.all)
        {
        }
        Body() = default;

        const uint32_t track_id_;
        const uint32_t default_sample_duration_;
        uint32_t default_sample_size_;
        uint32_t default_sample_flags_;
    } body_;
};

/// Movie Fragment Header
class MFHD_Atom : public Mp4VersionedAtom
{
public:
    MFHD_Atom(uint32_t sequence_number)
        : Mp4VersionedAtom("mfhd", 0, 0)
        , body_(sequence_number)
    {
        UpdateSize();
    }

    virtual uint32_t GetBodySize() const override
    {
        return sizeof(Body);
    }

    virtual void WriteAtom(std::vector<uint8_t> &buffer) override
    {
        WriteAtomHeader(buffer);
        WriteValue(buffer, body_.sequence_number_);
        WriteAtomChildren(buffer);
    }

    void SetSequenceNumber(uint32_t sequence_number)
    {
        body_.sequence_number_ = sequence_number;
    }

private:
    struct __attribute__((__packed__)) Body
    {
        explicit Body(uint32_t sequence_number)
            : sequence_number_(sequence_number)
        {
        }
        Body() = delete;

        uint32_t sequence_number_;
    } body_;
};

/// Chunk Offset Box
class STCO_Atom : public Mp4VersionedAtom
{
public:
    STCO_Atom()
        : Mp4VersionedAtom("stco", 0, 0)
    {
        UpdateSize();
    }

    virtual uint32_t GetBodySize() const override
    {
        return 4;
    }

    virtual void WriteAtom(std::vector<uint8_t> &buffer) override
    {
        WriteAtomHeader(buffer);
        WriteValue32(buffer, 0); // number of entries
    }
};

/// Decoding Time to Sample Box
class STTS_Atom : public Mp4VersionedAtom
{
public:
    STTS_Atom()
        : Mp4VersionedAtom("stts", 0, 0)
    {
        UpdateSize();
    }

    virtual uint32_t GetBodySize() const override
    {
        return 4;
    }

    virtual void WriteAtom(std::vector<uint8_t> &buffer) override
    {
        WriteAtomHeader(buffer);
        WriteValue32(buffer, 0); // number of entries
    }
};

/// Sample To Chunk Box
class STSC_Atom : public Mp4VersionedAtom
{
public:
    STSC_Atom()
        : Mp4VersionedAtom("stsc", 0, 0)
    {
        UpdateSize();
    }

    virtual uint32_t GetBodySize() const override
    {
        return 4;
    }

    virtual void WriteAtom(std::vector<uint8_t> &buffer) override
    {
        WriteAtomHeader(buffer);
        WriteValue32(buffer, 0); // number of entries
    }
};

/// Sample Size Box
class STSZ_Atom : public Mp4VersionedAtom
{
public:
    STSZ_Atom()
        : Mp4VersionedAtom("stsc", 0, 0)
    {
        UpdateSize();
    }

    virtual uint32_t GetBodySize() const override
    {
        return 8;
    }

    virtual void WriteAtom(std::vector<uint8_t> &buffer) override
    {
        WriteAtomHeader(buffer);
        WriteValue32(buffer, 0); // sample size
        WriteValue32(buffer, 0); // number of sample size entries
    }
};

/// Sample Description Box
class STSD_Atom : public Mp4VersionedAtom
{
public:
    STSD_Atom()
        : Mp4VersionedAtom("stsd", 0, 0)
    {
        UpdateSize();
    }

    virtual uint32_t GetBodySize() const override
    {
        return 4;
    }

    virtual void WriteAtom(std::vector<uint8_t> &buffer) override
    {
        WriteAtomHeader(buffer);
        WriteValue32(buffer, static_cast<uint32_t>(children_.size()));
        WriteAtomChildren(buffer);
    }
};

class AVC1_Atom : public Mp4Atom
{
public:
    AVC1_Atom(uint16_t width, uint16_t height)
        : Mp4Atom("avc1")
        , body_(width, height)
    {
        UpdateSize();
    }

    virtual uint32_t GetBodySize() const override
    {
        return sizeof(Body);
    }

    virtual void WriteAtom(std::vector<uint8_t> &buffer) override
    {
        WriteAtomHeader(buffer);
        WriteValue(buffer, body_.reserved0_);
        WriteValue(buffer, body_.reserved1_);
        WriteValue(buffer, body_.data_reference_index_);
        WriteValue(buffer, body_.video_encoding_version_);
        WriteValue(buffer, body_.video_encoding_revision_level_);
        WriteValue(buffer, body_.video_encoding_vendor_);
        WriteValue(buffer, body_.video_temporal_quality_);
        WriteValue(buffer, body_.video_spatial_quality_);
        WriteValue(buffer, body_.width_);
        WriteValue(buffer, body_.height_);
        WriteValue(buffer, body_.dpi_horiz_);
        WriteValue(buffer, body_.dpi_vert_);
        WriteValue(buffer, body_.video_data_size_);
        WriteValue(buffer, body_.video_frame_count_);
        WriteString(buffer, "AVC Coding", 32);
        WriteValue(buffer, body_.video_pixel_depth_);
        WriteValue(buffer, body_.video_color_table_id_);

        WriteAtomChildren(buffer);
    }

private:
    struct __attribute__((__packed__)) Body
    {
        explicit Body(uint16_t width, uint16_t height)
            : width_(width)
            , height_(height)
        {
        }
        Body() = delete;

        const uint32_t reserved0_                     = 0;
        const uint16_t reserved1_                     = 0;
        const uint16_t data_reference_index_          = 1;
        const uint16_t video_encoding_version_        = 0;
        const uint16_t video_encoding_revision_level_ = 0;
        const uint32_t video_encoding_vendor_         = 0;
        const uint32_t video_temporal_quality_        = 0;
        const uint32_t video_spatial_quality_         = 0;
        uint16_t width_;
        uint16_t height_;
        const uint32_t dpi_horiz_            = 0x00480000;
        const uint32_t dpi_vert_             = 0x00480000;
        const uint32_t video_data_size_      = 0;
        const uint16_t video_frame_count_    = 1;
        const char video_encoder_name_[32]   = "AVC Coding";
        const uint16_t video_pixel_depth_    = 24;
        const uint16_t video_color_table_id_ = 0xFFFF;
    } body_;
};

class AVCC_Atom : public Mp4Atom
{
public:
    AVCC_Atom(const std::vector<uint8_t> &sps, const std::vector<uint8_t> &pps)
        : Mp4Atom("avcC")
        , sps_(sps)
        , pps_(pps)
    {
        UpdateSize();
    }

    virtual uint32_t GetBodySize() const override
    {
        return 11 + static_cast<uint32_t>(sps_.size() + pps_.size());
    }

    virtual void WriteAtom(std::vector<uint8_t> &buffer) override
    {
        WriteAtomHeader(buffer);
        WriteValue8(buffer, 1);
        WriteValue8(buffer, sps_[1]); // profile
        WriteValue8(buffer, sps_[2]); // compatibility
        WriteValue8(buffer, sps_[3]); // level
        WriteValue8(buffer, 0xFF);    // Nalu length -1 (4)
        WriteValue8(buffer, 0xE1);    // Number of sps (1)
        WriteValue16(buffer, static_cast<uint16_t>(sps_.size()));

        WriteData(buffer, sps_.begin(), sps_.end());
        WriteValue8(buffer, 1);                                   // number of pps segments
        WriteValue16(buffer, static_cast<uint16_t>(pps_.size())); // size pps segment
        WriteData(buffer, pps_.begin(), pps_.end());
    }

private:
    std::vector<uint8_t> sps_;
    std::vector<uint8_t> pps_;
};

/// Video Media Header
class VMHD_Atom : public Mp4VersionedAtom
{
public:
    VMHD_Atom()
        : Mp4VersionedAtom("vmhd", 0, 0x000001)
    {
        UpdateSize();
    }

    virtual uint32_t GetBodySize() const override
    {
        return sizeof(uint64_t);
    }

    virtual void WriteAtom(std::vector<uint8_t> &buffer) override
    {
        WriteAtomHeader(buffer);
        WriteValue64(buffer, 0);
        WriteAtomChildren(buffer);
    }
};

/// Data Reference Box
class DREF_Atom : public Mp4VersionedAtom
{
public:
    DREF_Atom()
        : Mp4VersionedAtom("dref", 0, 0)
    {
        UpdateSize();
    }

    virtual uint32_t GetBodySize() const override
    {
        return sizeof(uint32_t);
    }

    virtual void WriteAtom(std::vector<uint8_t> &buffer) override
    {
        WriteAtomHeader(buffer);
        WriteValue32(buffer, static_cast<uint32_t>(children_.size()));
        WriteAtomChildren(buffer);
    }
};

class URL_Atom : public Mp4VersionedAtom
{
public:
    URL_Atom(const std::string &url)
        : Mp4VersionedAtom("url ", 0, 1)
    {
        UpdateSize();
    }

    virtual uint32_t GetBodySize() const override
    {
        return GetStringSize(url_);
    }

    virtual void WriteAtom(std::vector<uint8_t> &buffer) override
    {
        WriteAtomHeader(buffer);
        if (url_.size())
        {
            WriteString(buffer, url_);
        }
        WriteAtomChildren(buffer);
    }

private:
    std::string url_;
};

class URN_Atom : public Mp4VersionedAtom
{
public:
    URN_Atom(const std::string &name, const std::string &url)
        : Mp4VersionedAtom("urn ", 0, 0)
        , name_(name)
        , url_(url)
    {
        UpdateSize();
    }

    virtual uint32_t GetBodySize() const override
    {
        return GetStringSize(name_) + GetStringSize(url_);
    }

    virtual void WriteAtom(std::vector<uint8_t> &buffer) override
    {
        WriteAtomHeader(buffer);
        WriteString(buffer, name_);
        WriteString(buffer, url_);
    }

private:
    std::string name_;
    std::string url_;
};

/// Handler Reference Box
class HDLR_Atom : public Mp4VersionedAtom
{
public:
    HDLR_Atom(const std::string &handler_name)
        : Mp4VersionedAtom("hdlr", 0, 0)
        , handler_name_(handler_name)
    {
        UpdateSize();
    }

    virtual uint32_t GetBodySize() const override
    {
        return GetStringSize(handler_name_) + sizeof(Body);
    }

    virtual void WriteAtom(std::vector<uint8_t> &buffer) override
    {
        WriteAtomHeader(buffer);
        WriteValue(buffer, body_.type_);
        WriteValue(buffer, body_.media_type_);
        WriteValue(buffer, body_.manufacturer_reserved_);
        WriteValue(buffer, body_.component_reserved_flags_);
        WriteValue(buffer, body_.component_reserved_flags_mask_);
        WriteString(buffer, handler_name_);
    }

private:
    struct __attribute__((__packed__)) Body
    {
        const uint32_t type_                          = 0;
        const uint32_t media_type_                    = CreateAtomTag("vide");
        const uint32_t manufacturer_reserved_         = 0;
        const uint32_t component_reserved_flags_      = 0;
        const uint32_t component_reserved_flags_mask_ = 0;
    } body_;
    const std::string handler_name_;
};

/// Media Header Box
class MDHD_Atom : public Mp4VersionedAtom
{
public:
    MDHD_Atom(float fps)
        : Mp4VersionedAtom("mdhd", 1, 0)
        , body_(fps)
    {
        UpdateSize();
    }

    virtual uint32_t GetBodySize() const override
    {
        return sizeof(Body);
    }

    virtual void WriteAtom(std::vector<uint8_t> &buffer) override
    {
        WriteAtomHeader(buffer);
        WriteValue(buffer, body_.creation_time_);
        WriteValue(buffer, body_.modification_time_);
        WriteValue(buffer, body_.time_scale_);
        WriteValue(buffer, body_.duration_);
        WriteValue(buffer, body_.content_language_);
        WriteValue(buffer, body_.quality_);
    }

private:
    struct __attribute__((__packed__)) Body
    {
        explicit Body(float fps)
            : time_scale_(std::max(1u, static_cast<uint32_t>(fps + 0.5f)))
        {
            const char *language  = "und"; // Undetermined
            uint16_t language_enc = 0;
            for (int idx = 0; idx < 3; idx++)
            {
                language_enc = language_enc << 5;
                language_enc |= language[idx] - 0x60;
            }

            content_language_ = language_enc;
        }
        Body() = delete;

        const uint64_t creation_time_     = 0;
        const uint64_t modification_time_ = 0;
        // number of time units that pass in one second
        const uint32_t time_scale_;
        // duartion must be zero for fragmented MPEG-4
        const uint64_t duration_ = 0;
        uint16_t content_language_;
        uint16_t quality_ = 0;
    } body_;
};

/// Track Header Box
class TKHD_Atom : public Mp4VersionedAtom
{
public:
    static const uint32_t TRACK_ENABLED_C    = 0x1;
    static const uint32_t TRACK_IN_VIDEO_C   = 0x2;
    static const uint32_t TRACK_IN_PREVIEW_C = 0x4;
    TKHD_Atom(uint32_t track_id, uint16_t width, uint16_t height)
        : Mp4VersionedAtom("tkhd", 1, TRACK_ENABLED_C | TRACK_IN_VIDEO_C | TRACK_IN_PREVIEW_C)
        , body_(track_id, width, height)
    {
        UpdateSize();
    }

    virtual uint32_t GetBodySize() const override
    {
        return sizeof(Body);
    }

    virtual void WriteAtom(std::vector<uint8_t> &buffer) override
    {
        WriteAtomHeader(buffer);
        WriteValue(buffer, body_.creation_time_);
        WriteValue(buffer, body_.modification_time_);

        WriteValue(buffer, body_.track_ID_);
        WriteValue(buffer, body_.reserved0_);
        WriteValue(buffer, body_.reserved1_);
        WriteValue(buffer, body_.duration_);
        WriteValue(buffer, body_.reserved2_);
        WriteValue(buffer, body_.layer_);
        WriteValue(buffer, body_.alternate_other_);
        WriteValue(buffer, body_.volume);
        WriteValue(buffer, body_.reserved3_);

        WriteValue(buffer, body_.matrix_A_);
        WriteValue(buffer, body_.matrix_B_);
        WriteValue(buffer, body_.matrix_U_);
        WriteValue(buffer, body_.matrix_C_);
        WriteValue(buffer, body_.matrix_D_);
        WriteValue(buffer, body_.matrix_V_);
        WriteValue(buffer, body_.matrix_X_);
        WriteValue(buffer, body_.matrix_Y_);
        WriteValue(buffer, body_.matrix_W_);

        WriteValue(buffer, body_.width_);
        WriteValue(buffer, body_.height_);
    }

private:
    struct __attribute__((__packed__)) Body
    {
        explicit Body(uint32_t track_id, uint16_t width, uint16_t height)
            : track_ID_(track_id)
            , width_(width << 16)
            , height_(height << 16)
        {
        }
        Body() = delete;

        const uint64_t creation_time_     = 0;
        const uint64_t modification_time_ = 0;
        const uint32_t track_ID_;
        const uint32_t reserved0_ = 0;
        // duartion must be zero for fragmented MPEG-4
        const uint64_t duration_        = 0;
        const uint32_t reserved1_       = 0;
        const uint32_t reserved2_       = 0;
        const uint16_t layer_           = 0;
        const uint16_t alternate_other_ = 0;
        const uint16_t volume           = 0;
        const uint16_t reserved3_       = 0;
        const uint32_t matrix_A_        = 0x00010000;
        const uint32_t matrix_B_        = 0x00000000;
        const uint32_t matrix_U_        = 0x00000000;
        const uint32_t matrix_C_        = 0x00000000;
        const uint32_t matrix_D_        = 0x00010000;
        const uint32_t matrix_V_        = 0x00000000;
        const uint32_t matrix_X_        = 0x00000000;
        const uint32_t matrix_Y_        = 0x00000000;
        const uint32_t matrix_W_        = 0x40000000;
        const uint32_t width_;
        const uint32_t height_;
    } body_;
};

/// Track Extends Box
class TREX_Atom : public Mp4VersionedAtom
{
public:
    TREX_Atom(uint32_t track_id)
        : Mp4VersionedAtom("trex", 0, 0)
        , body_(track_id)
    {
        UpdateSize();
    }

    virtual uint32_t GetBodySize() const override
    {
        return sizeof(Body);
    }

    virtual void WriteAtom(std::vector<uint8_t> &buffer) override
    {
        WriteAtomHeader(buffer);

        WriteValue(buffer, body_.track_id_);
        WriteValue(buffer, body_.sample_description_index_);
        WriteValue(buffer, body_.sample_duration_);
        WriteValue(buffer, body_.sample_size_);
        WriteValue(buffer, body_.sample_flags_);
    }

private:
    struct __attribute__((__packed__)) Body
    {
        explicit Body(uint32_t track_id)
            : track_id_(track_id)
        {
        }
        Body() = delete;

        const uint32_t track_id_;
        const uint32_t sample_description_index_ = 1;
        const uint32_t sample_duration_          = 0;
        const uint32_t sample_size_              = 0;
        const uint32_t sample_flags_             = 0;
    } body_;
};

/// Movie Extends Header Box
class MEHD_Atom : public Mp4VersionedAtom
{
public:
    MEHD_Atom(uint32_t fragment_duration)
        : Mp4VersionedAtom("mehd", 0, 0)
        , body_(fragment_duration)
    {
        UpdateSize();
    }

    virtual uint32_t GetBodySize() const override
    {
        return sizeof(Body);
    }

    virtual void WriteAtom(std::vector<uint8_t> &buffer) override
    {
        WriteAtomHeader(buffer);
        WriteValue(buffer, body_.fragment_duration_);
    }

private:
    struct __attribute__((__packed__)) Body
    {
        explicit Body(uint32_t fragment_duration)
            : fragment_duration_(fragment_duration)
        {
        }
        Body() = delete;

        const uint32_t fragment_duration_;
    } body_;
};

/// Movie Header Box
class MVHD_Atom : public Mp4VersionedAtom
{
public:
    MVHD_Atom(uint32_t track_id)
        : Mp4VersionedAtom("mvhd", 1, 0)
        , body_(track_id)
    {
        UpdateSize();
    }

    virtual uint32_t GetBodySize() const override
    {
        return sizeof(Body);
    }

    virtual void WriteAtom(std::vector<uint8_t> &buffer) override
    {
        WriteAtomHeader(buffer);

        WriteValue(buffer, body_.creation_time);
        WriteValue(buffer, body_.modification_time);
        WriteValue(buffer, body_.time_scale_);
        WriteValue(buffer, body_.duration_);
        WriteValue(buffer, body_.rate_);
        WriteValue(buffer, body_.volume_);

        WriteValue(buffer, body_.reserved0_);
        WriteValue(buffer, body_.reserved1_);
        WriteValue(buffer, body_.reserved2_);

        WriteValue(buffer, body_.matrix_A_);
        WriteValue(buffer, body_.matrix_B_);
        WriteValue(buffer, body_.matrix_U_);
        WriteValue(buffer, body_.matrix_C_);
        WriteValue(buffer, body_.matrix_D_);
        WriteValue(buffer, body_.matrix_V_);
        WriteValue(buffer, body_.matrix_X_);
        WriteValue(buffer, body_.matrix_Y_);
        WriteValue(buffer, body_.matrix_W_);

        WriteValue(buffer, body_.reserved3_);
        WriteValue(buffer, body_.reserved4_);
        WriteValue(buffer, body_.reserved5_);
        WriteValue(buffer, body_.reserved6_);
        WriteValue(buffer, body_.reserved7_);
        WriteValue(buffer, body_.reserved8_);

        WriteValue(buffer, body_.next_track_ID_);
    }

private:
    struct __attribute__((__packed__)) Body
    {
        explicit Body(uint32_t track_id)
            : next_track_ID_(track_id + 1)
        {
        }
        Body() = delete;

        const uint64_t creation_time     = 0;
        const uint64_t modification_time = 0;
        // number of time units that pass in one second
        const uint32_t time_scale_ = 1000;
        // duartion must be zero for fragmented MPEG-4
        const uint64_t duration_ = 0;
        // 16.16 fixed point preferred rate
        const uint32_t rate_ = 0x00010000;
        // 8.8 fixed-point preferred volume
        const uint16_t volume_ = 0x0100;

        const uint16_t reserved0_ = 0;
        const uint32_t reserved1_ = 0;
        const uint32_t reserved2_ = 0;

        const uint32_t matrix_A_ = 0x00010000;
        const uint32_t matrix_B_ = 0x00000000;
        const uint32_t matrix_U_ = 0x00000000;
        const uint32_t matrix_C_ = 0x00000000;
        const uint32_t matrix_D_ = 0x00010000;
        const uint32_t matrix_V_ = 0x00000000;
        const uint32_t matrix_X_ = 0x00000000;
        const uint32_t matrix_Y_ = 0x00000000;
        const uint32_t matrix_W_ = 0x40000000;

        const uint32_t reserved3_ = 0;
        const uint32_t reserved4_ = 0;
        const uint32_t reserved5_ = 0;
        const uint32_t reserved6_ = 0;
        const uint32_t reserved7_ = 0;
        const uint32_t reserved8_ = 0;
        const uint32_t next_track_ID_;

    } body_;
};

/// File Type Box
class FTYP_Atom : public Mp4Atom
{
public:
    FTYP_Atom()
        : Mp4Atom("ftyp")
    {
        UpdateSize();
    }

    virtual uint32_t GetBodySize() const override
    {
        return sizeof(Body);
    }

    virtual void WriteAtom(std::vector<uint8_t> &buffer) override
    {
        WriteAtomHeader(buffer);
        WriteValue(buffer, body_.major_brand_);
        WriteValue(buffer, body_.minor_version_);
        WriteValue(buffer, body_.compatible_brands0_);
        WriteValue(buffer, body_.compatible_brands1_);
        WriteValue(buffer, body_.compatible_brands2_);
        WriteValue(buffer, body_.compatible_brands3_);
        WriteValue(buffer, body_.compatible_brands4_);
    }

private:
    struct __attribute__((__packed__)) Body
    {
        const uint32_t major_brand_        = CreateAtomTag("mp42");
        const uint32_t minor_version_      = 0x0200;
        const uint32_t compatible_brands0_ = CreateAtomTag("mp41");
        const uint32_t compatible_brands1_ = CreateAtomTag("avc1");
        const uint32_t compatible_brands2_ = CreateAtomTag("isom");
        const uint32_t compatible_brands3_ = CreateAtomTag("iso2");
        const uint32_t compatible_brands4_ = CreateAtomTag("iso5");
    } body_;
};

class MP4Wrapper::Impl
{

public:
    Impl();

    /**
     * Reset the stream, this will write the initializations segment again.
     */
    void ResetStream();

    /**
     * Wrap a video frame.
     *
     * @param width [in] stream width
     * @param height [in] stream height
     * @param fps [in] stream frames per second
     * @param type [in] input frame type
     * @param input_frame [in] input frame
     * @param output_buffer [in] stream output buffer
     */
    void Wrap(uint32_t width, uint32_t height, float fps, Type type, const std::vector<uint8_t> &input_frame,
              std::vector<uint8_t> &output_buffer);

private:
    void WriteInitializationSegment(std::vector<uint8_t> &output_buffer);
    void FindNalus(Type type, const std::vector<uint8_t>::const_iterator begin,
                   const std::vector<uint8_t>::const_iterator end);

    // values set by FindNalus()
    std::vector<Nalu> nalus_;
    std::vector<uint8_t> sps_nalu_;
    std::vector<uint8_t> pps_nalu_;
    size_t new_size_ = 0;

    // Atoms that need to be updated every frame
    std::shared_ptr<Mp4Atom> moof_atom_;
    std::shared_ptr<TFHD_Atom> tfhd_atom_;
    std::shared_ptr<TFDT_Atom> tfdt_atom_;
    std::shared_ptr<TRUN_Atom> trun_atom_;
    std::shared_ptr<MDAT_Atom> mdat_atom_;

    uint32_t track_id_ = 1;
    uint32_t seqno_    = 0;
    uint32_t width_    = 0;
    uint32_t height_   = 0;
    float fps_         = 0.f;
};

void MP4Wrapper::Impl::FindNalus(Type type, const std::vector<uint8_t>::const_iterator begin,
                                 const std::vector<uint8_t>::const_iterator end)
{
    nalus_.clear();
    sps_nalu_.clear();
    pps_nalu_.clear();
    new_size_ = 0;

    size_t startcodesize = 4u;

    auto startcodebegin = begin;
    auto sectionbegin   = begin;

    while (sectionbegin < end)
    {
        static const std::vector<uint8_t> twozeros{0x00, 0x00};
        const auto next = std::search(sectionbegin + startcodesize, end, twozeros.begin(), twozeros.end());

        size_t nextstartcodesize = 0u;

        if (next != end)
        {
            // emulation byte
            if (next + 2u < end && *(next + 2u) == 0x03)
            {
                sectionbegin = next + 3u;
                continue;
            }

            if (next + 2u < end && *(next + 2u) == 0x01)
            {
                //3 byte Annex B startcode
                nextstartcodesize = 3u;
            }
            else if (next + 3u < end && *(next + 2u) == 0x00 && *(next + 3u) == 0x01)
            {
                //4 byte Annex B startcode
                nextstartcodesize = 4u;
            }
            else
            {
                //nothing to see here after all, move along
                sectionbegin = next + 1;
                continue;
            }
        }

        //store NALU description
        Nalu nalu;

        nalu.nalubegin_ = startcodebegin + startcodesize;
        nalu.naluend_   = next;
        nalus_.push_back(nalu);
        new_size_ += 4 + (nalu.naluend_ - nalu.nalubegin_);

        if (((type == Type::H264) && (nalu.nalutypeh264() == Nalu::H264_SPS_TYPE)) ||
            ((type == Type::HEVC) && (nalu.nalutypehevc() == Nalu::HEVC_SPS_TYPE)))
        {
            if (sps_nalu_.empty())
            {
                sps_nalu_.insert(sps_nalu_.end(), nalu.nalubegin_, nalu.naluend_);
            }
        }
        if (((type == Type::H264) && (nalu.nalutypeh264() == Nalu::H264_PPS_TYPE)) ||
            ((type == Type::HEVC) && (nalu.nalutypehevc() == Nalu::HEVC_PPS_TYPE)))
        {
            if (pps_nalu_.empty())
            {
                pps_nalu_.insert(pps_nalu_.end(), nalu.nalubegin_, nalu.naluend_);
            }
        }

        startcodesize  = nextstartcodesize;
        startcodebegin = next;
        sectionbegin   = next;
    } // namespace clara::viz
}

MP4Wrapper::Impl::Impl() {}

void MP4Wrapper::Impl::WriteInitializationSegment(std::vector<uint8_t> &output_buffer)
{
    // create the atoms which need to be updated for each frame
    moof_atom_                         = std::make_shared<Mp4Atom>("moof");
    std::shared_ptr<Mp4Atom> mfhd_atom = std::make_shared<MFHD_Atom>(seqno_);
    moof_atom_->AddChild(mfhd_atom);

    std::shared_ptr<Mp4Atom> traf_atom = std::make_shared<Mp4Atom>("traf");

    SampleFlags default_sample_flags;
    default_sample_flags.sample_is_non_sync_sample = 1;
    default_sample_flags.sample_depends_on         = SAMPLE_FLAGS_DEPENDS_ON_OTHERS;
    tfhd_atom_ = std::make_shared<TFHD_Atom>(track_id_, /*default_sample_duration*/ 1, /*default_sample_size*/ 0,
                                             default_sample_flags);
    traf_atom->AddChild(tfhd_atom_);

    tfdt_atom_ = std::make_shared<TFDT_Atom>(seqno_);
    traf_atom->AddChild(tfdt_atom_);

    SampleFlags first_sample_flags;
    first_sample_flags.all               = 0;
    first_sample_flags.sample_depends_on = SAMPLE_FLAGS_DEPENDS_ON_NO_OTHERS;
    trun_atom_ = std::make_shared<TRUN_Atom>(/*data_offset*/ 8, /*sample_size*/ 0, first_sample_flags);
    traf_atom->AddChild(trun_atom_);

    moof_atom_->AddChild(traf_atom);
    mdat_atom_ = std::make_shared<MDAT_Atom>();

    // now write the initialization segment
    std::shared_ptr<FTYP_Atom> ftyp_atom = std::make_shared<FTYP_Atom>();
    ftyp_atom->WriteAtom(output_buffer);

    std::shared_ptr<Mp4Atom> moov_atom = std::make_shared<Mp4Atom>("moov");
    // movie header
    std::shared_ptr<Mp4Atom> mvhd_atom = std::make_shared<MVHD_Atom>(track_id_);
    moov_atom->AddChild(mvhd_atom);

    std::shared_ptr<Mp4Atom> trak_atom = std::make_shared<Mp4Atom>("trak");
    std::shared_ptr<Mp4Atom> tkhd_atom = std::make_shared<TKHD_Atom>(track_id_, width_, height_);
    trak_atom->AddChild(tkhd_atom);

    std::shared_ptr<Mp4Atom> mdia_atom = std::make_shared<Mp4Atom>("mdia");
    std::shared_ptr<Mp4Atom> mdhd_atom = std::make_shared<MDHD_Atom>(fps_);
    std::shared_ptr<Mp4Atom> hdlr_atom = std::make_shared<HDLR_Atom>("NVIDIA MPEG4 container");
    mdia_atom->AddChild(mdhd_atom);
    mdia_atom->AddChild(hdlr_atom);

    std::shared_ptr<Mp4Atom> dref_atom = std::make_shared<DREF_Atom>();
    std::shared_ptr<Mp4Atom> url_atom  = std::make_shared<URL_Atom>("");
    dref_atom->AddChild(url_atom);

    std::shared_ptr<Mp4Atom> dinf_atom = std::make_shared<Mp4Atom>("dinf");
    dinf_atom->AddChild(dref_atom);

    std::shared_ptr<Mp4Atom> minf_atom = std::make_shared<Mp4Atom>("minf");
    std::shared_ptr<Mp4Atom> vmhd_atom = std::make_shared<VMHD_Atom>();
    minf_atom->AddChild(vmhd_atom);
    minf_atom->AddChild(dinf_atom);

    std::shared_ptr<Mp4Atom> stbl_atom = std::make_shared<Mp4Atom>("stbl");
    std::shared_ptr<Mp4Atom> stsd_atom = std::make_shared<STSD_Atom>();
    std::shared_ptr<Mp4Atom> stsz_atom = std::make_shared<STSZ_Atom>();
    std::shared_ptr<Mp4Atom> stsc_atom = std::make_shared<STSC_Atom>();
    std::shared_ptr<Mp4Atom> stts_atom = std::make_shared<STTS_Atom>();
    std::shared_ptr<Mp4Atom> stco_atom = std::make_shared<STCO_Atom>();
    std::shared_ptr<Mp4Atom> avc1_atom = std::make_shared<AVC1_Atom>(width_, height_);
    if (!sps_nalu_.empty() && !pps_nalu_.empty())
    {
        std::shared_ptr<Mp4Atom> avcC_atom = std::make_shared<AVCC_Atom>(sps_nalu_, pps_nalu_);
        avc1_atom->AddChild(avcC_atom);
    }

    stsd_atom->AddChild(avc1_atom);
    stbl_atom->AddChild(stsd_atom);
    stbl_atom->AddChild(stsz_atom);
    stbl_atom->AddChild(stsc_atom);
    stbl_atom->AddChild(stts_atom);
    stbl_atom->AddChild(stco_atom);

    minf_atom->AddChild(stbl_atom);
    mdia_atom->AddChild(minf_atom);
    trak_atom->AddChild(mdia_atom);
    moov_atom->AddChild(trak_atom);

    std::shared_ptr<Mp4Atom> mvex_atom = std::make_shared<Mp4Atom>("mvex");
    std::shared_ptr<Mp4Atom> mehd_atom = std::make_shared<MEHD_Atom>(0);
    std::shared_ptr<Mp4Atom> trex_atom = std::make_shared<TREX_Atom>(track_id_);

    mvex_atom->AddChild(mehd_atom);
    mvex_atom->AddChild(trex_atom);
    moov_atom->AddChild(mvex_atom);

    moov_atom->WriteAtom(output_buffer);
}

void MP4Wrapper::Impl::ResetStream()
{
    // forcing width, height and fps_ to zero will reset the stream
    width_  = 0;
    height_ = 0;
    fps_    = 0.f;
}

void MP4Wrapper::Impl::Wrap(uint32_t width, uint32_t height, float fps, Type type,
                            const std::vector<uint8_t> &inputFrame, std::vector<uint8_t> &output_buffer)
{
    FindNalus(type, inputFrame.begin(), inputFrame.end());

    if ((width != width_) || (height != height_) || (fps != fps_))
    {
        width_  = width;
        height_ = height;
        fps_    = fps;
        seqno_  = 0;

        // write initialization segment if frame size changed
        WriteInitializationSegment(output_buffer);
    }

    // Encode frame
    uint32_t moof_size = moof_atom_->GetSize();

    // Update atoms
    tfdt_atom_->SetBaseMediaDecodeTime(seqno_++);
    trun_atom_->SetDataOffset(static_cast<uint32_t>(moof_size + 8));
    trun_atom_->SetSampleSize(static_cast<uint32_t>(new_size_));

    mdat_atom_->SetNalus(&(nalus_), new_size_);
    tfhd_atom_->SetDefaultSampleSize(static_cast<uint32_t>(new_size_));

    output_buffer.reserve(output_buffer.size() + moof_size + mdat_atom_->GetSize());

    moof_atom_->WriteAtom(output_buffer);
    mdat_atom_->WriteAtom(output_buffer);
}

MP4Wrapper::MP4Wrapper()
    : impl_(new Impl)
{
}

MP4Wrapper::~MP4Wrapper() {}

void MP4Wrapper::ResetStream()
{
    impl_->ResetStream();
}

void MP4Wrapper::Wrap(uint32_t width, uint32_t height, float fps, Type type, const std::vector<uint8_t> &input_frame,
                      std::vector<uint8_t> &output_buffer)
{
    if (width == 0)
    {
        throw InvalidArgument("width") << "is zero";
    }
    if (height == 0)
    {
        throw InvalidArgument("height") << "is zero";
    }
    if (fps < 0.f)
    {
        throw InvalidArgument("fps") << "is negative";
    }
    if (input_frame.empty())
    {
        throw InvalidArgument("input_frame") << "is empty";
    }

    impl_->Wrap(width, height, fps, type, input_frame, output_buffer);
}

} // namespace clara::viz
