/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION. All rights reserved.
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

#include <array>
#include <cassert>
#include <cstdint>
#include <cmath>

#include "claraviz/util/VectorT.h"

namespace clara::viz
{

#ifdef __CUDACC__
#define CUDA_FUNC __host__ __device__
#else
#define CUDA_FUNC
#endif

/**
 * A float matrix template class (row major)
 */
template<uint32_t ROWS, uint32_t COLUMNS>
class MatrixT
{
public:
    /**
     * Construct (initialized to identity)
     */
    MatrixT()
    {
        Identity();
    }

    /**
     * Construct and set to value
     */
    explicit MatrixT(float value)
    {
        Set(value);
    }

    /**
     * Construct and set with std::array
     */
    explicit MatrixT(const std::array<std::array<float, COLUMNS>, ROWS> &array)
    {
        for (uint32_t row = 0; row < ROWS; ++row)
        {
            for (uint32_t col = 0; col < COLUMNS; ++col)
            {
                operator()(row, col) = array[row][col];
            }
        }
    }

    /**
     * Set to identity
     */
    void Identity()
    {
        for (uint32_t row = 0; row < ROWS; ++row)
        {
            for (uint32_t col = 0; col < COLUMNS; ++col)
            {
                operator()(row, col) = (row == col) ? 1.f : 0.f;
            }
        }
    }

    /**
     * Set the scale factors
     *
     * [in] scale: scale factors
     */
    void SetScale(const std::array<float, (ROWS < COLUMNS) ? ROWS : COLUMNS> &scale)
    {
        for (uint32_t i = 0; i < scale.size(); ++i)
        {
            operator()(i, i) = scale[i];
        }
    }

    /**
     * Set the translation values
     *
     * @param[in] translate: translation values
     */
    void SetTranslate(const std::array<float, ROWS> &translate)
    {
        for (uint32_t row = 0; row < translate.size(); ++row)
        {
            operator()(row, COLUMNS - 1) = translate[row];
        }
    }

    /**
     * Set to value
     */
    void Set(float value)
    {
        for (uint32_t row = 0; row < ROWS; ++row)
        {
            for (uint32_t col = 0; col < COLUMNS; ++col)
            {
                operator()(row, col) = value;
            }
        }
    }

    /**
     * @returns true if all components of this == rhs
     */
    bool operator==(const MatrixT &rhs) const
    {
        bool result = true;
        for (uint32_t i = 0; i < ROWS; ++i)
        {
            for (uint32_t j = 0; j < COLUMNS; ++j)
            {
                result &= (values_[i][j] == rhs.values_[i][j]);
            }
        }
        return result;
    }

    /**
     * @returns true if any component of this != rhs
     */
    bool operator!=(const MatrixT &rhs) const
    {
        return !(*this == rhs);
    }

    /**
     * matrix *= matrix
     */
    MatrixT &operator*=(const MatrixT &rhs)
    {
        MatrixT copy(*this);

        Set(0.f);

        for (uint32_t i = 0; i < ROWS; ++i)
        {
            for (uint32_t j = 0; j < COLUMNS; ++j)
            {
                for (uint32_t k = 0; k < std::min(ROWS, COLUMNS); ++k)
                {
                    operator()(i, j) += copy(i, k) * rhs(k, j);
                }
                // handle non-square matrices
                for (uint32_t k = std::min(ROWS, COLUMNS); k < std::max(ROWS, COLUMNS); ++k)
                {
                    operator()(i, j) += ((k < COLUMNS) ? copy(i, k)
                                         : (i == k)    ? 1.f
                                                       : 0.f) *
                                        ((k < ROWS) ? rhs(k, j)
                                         : (j == k) ? 1.f
                                                    : 0.f);
                }
            }
        }
        return *this;
    }

    /**
     * matrix * matrix
     */
    friend MatrixT operator*(const MatrixT &lhs, const MatrixT &rhs)
    {
        MatrixT result(lhs);

        result *= rhs;

        return result;
    }

    /**
     * vector = matrix * vector
     */
    template<typename T, uint32_t VROWS>
    CUDA_FUNC VectorT<T, (ROWS > VROWS) ? ROWS : VROWS> operator*(const VectorT<T, VROWS> &src) const
    {
        constexpr auto RROWS = (ROWS > VROWS) ? ROWS : VROWS;
        VectorT<float, RROWS> result(0.f);

        for (uint32_t i = 0; i < RROWS; ++i)
        {
            for (uint32_t j = 0; j < ((VROWS > COLUMNS) ? VROWS : COLUMNS); ++j)
            {
                result(i) += ((j < VROWS) ? src(j) : 1.f) *
                             (((i < ROWS) && (j < COLUMNS)) ? operator()(i, j) : ((i == j) ? 1.0f : 0.0f));
            }
        }

        return result;
    }

    /**
     * Access operator
     */
    CUDA_FUNC float &operator()(uint32_t row, uint32_t col)
    {
        assert((row < ROWS) && (col < COLUMNS));
        return values_[row][col];
    }

    /**
     * Access operator (const)
     */
    CUDA_FUNC const float &operator()(uint32_t row, uint32_t col) const
    {
        assert((row < ROWS) && (col < COLUMNS));
        return values_[row][col];
    }

    /**
     * Calculate the inverse
     */
    MatrixT Inverse() const
    {
        MatrixT inv;

        // create a square copy of the matrix with the unity matrix appended
        const auto SQUARE = std::max(ROWS, COLUMNS);
        float rows[SQUARE][SQUARE * 2];
        float *s[SQUARE];

        for (uint32_t i = 0; i < SQUARE; ++i)
            s[i] = &rows[i][0];

        // initialize to identity
        for (uint32_t i = 0; i < SQUARE; ++i)
        {
            for (uint32_t j = 0; j < SQUARE * 2; ++j)
            {
                s[i][j] = (i == j % SQUARE) ? 1.f : 0.f;
            }
        }

        // copy the matrix
        for (uint32_t i = 0; i < ROWS; ++i)
        {
            for (uint32_t j = 0; j < COLUMNS; ++j)
            {
                s[i][j] = operator()(i, j);
            }
        }

        float scp[SQUARE];
        for (uint32_t i = 0; i < SQUARE; ++i)
        {
            scp[i] = float(fabs(s[i][0]));
            for (uint32_t j = 1; j < SQUARE; ++j)
            {
                if (float(fabs(s[i][j])) > scp[i])
                    scp[i] = float(fabs(s[i][j]));
            }
            if (scp[i] == 0.f)
                return inv; // singular matrix!
        }

        for (uint32_t i = 0; i < SQUARE; ++i)
        {
            // select pivot row
            uint32_t pivotTo = i;
            float scpMax     = float(fabs(s[i][i] / scp[i]));
            // find out which row should be on top
            for (uint32_t p = i + 1; p < SQUARE; ++p)
            {
                if (float(fabs(s[p][i] / scp[p])) > scpMax)
                {
                    scpMax  = float(fabs(s[p][i] / scp[p]));
                    pivotTo = p;
                }
            }
            // Pivot if necessary
            if (pivotTo != i)
            {
                float *const tmpRow = s[i];
                s[i]                = s[pivotTo];
                s[pivotTo]          = tmpRow;
                const float tmpScp  = scp[i];
                scp[i]              = scp[pivotTo];
                scp[pivotTo]        = tmpScp;
            }

            // perform gaussian elimination
            for (uint32_t j = i + 1; j < SQUARE; ++j)
            {
                const float mji = s[j][i] / s[i][i];
                s[j][i]         = 0.f;
                for (uint32_t jj = i + 1; jj < 2 * SQUARE; ++jj)
                    s[j][jj] -= mji * s[i][jj];
            }
        }
        if (s[SQUARE - 1][SQUARE - 1] == 0.0)
            return inv; // singular matrix!

        //
        // Now we have an upper triangular matrix.
        //
        //  x x x x | y y y y
        //  0 x x x | y y y y
        //  0 0 x x | y y y y
        //  0 0 0 x | y y y y
        //
        //  we'll back substitute to get the inverse
        //
        //  1 0 0 0 | z z z z
        //  0 1 0 0 | z z z z
        //  0 0 1 0 | z z z z
        //  0 0 0 1 | z z z z
        //

        for (int32_t i = SQUARE - 1; i > 0; --i)
        {
            for (int32_t j = i - 1; j > -1; --j)
            {
                const float mij = s[j][i] / s[i][i];
                for (uint32_t jj = j + 1; jj < SQUARE * 2; ++jj)
                    s[j][jj] -= mij * s[i][jj];
            }
        }

        // copy to the result
        for (uint32_t i = 0; i < ROWS; ++i)
        {
            for (uint32_t j = 0; j < COLUMNS; ++j)
            {
                inv(i, j) = s[i][j + SQUARE] / s[i][i];
            }
        }

        return inv;
    }

private:
    float values_[ROWS][COLUMNS];
};

// define some commonly used matrix types
typedef MatrixT<3, 3> Matrix3x3;
typedef MatrixT<4, 4> Matrix4x4;

#undef CUDA_FUNC

} // namespace clara::viz
