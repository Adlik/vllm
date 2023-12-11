/*
 * Adapted from https://github.com/InternLM/lmdeploy
 * Copyright (c) OpenMMLab. All rights reserved.
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

#include "metric.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <memory>
#include <vector>

namespace vllm {

extern bool g_dump_kernel_info_once;

class GemmS4F16 {
public:
    GemmS4F16();

    ~GemmS4F16();

    enum Type
    {
        kGemm,
        kFusedSiluFfn
    };

    void Measure(half*                C,
                 const uint*          A,
                 const half*          B,
                 const half2*         Q,
                 int                  m,
                 int                  n,
                 int                  k,
                 int                  group_size,
                 Type                 type,
                 std::vector<Metric>& metrics,
                 cudaStream_t         st);

    void Run(half*        C,
             const uint*  A,
             const half*  B,
             const half2* Q,
             int          m,
             int          n,
             int          k,
             int          group_size,
             Type         type,
             int          algo_id,
             cudaStream_t st);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace vllm
