// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <fstream>
#include <include/postprocess_op.h>
#include <include/preprocess_op.h>
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <Net.h>

namespace PaddleOCR {
    class DBDetector {
    public:
        // Init DBDetector Configuration
        explicit DBDetector(const std::string &model_dir, const bool &use_gpu,
                            const int &gpu_id, const int &gpu_mem,
                            const int &cpu_math_library_num_threads,
                            const bool &use_mkldnn, const std::string &limit_type,
                            const int &limit_side_len, const double &det_db_thresh,
                            const double &det_db_box_thresh,
                            const double &det_db_unclip_ratio,
                            const std::string &det_db_score_mode,
                            const bool &use_dilation, const bool &use_tensorrt,
                            const std::string &precision) noexcept;

        // Load Paddle inference model
        void LoadModel(const std::string &model_dir) noexcept;

        // Run predictor
        void Run(const cv::Mat &img,
                 std::vector<std::vector<std::vector<int> > > &boxes,
                 std::vector<double> &times) noexcept;

    private:
        ncnn::Net predictor_;

        bool use_gpu_ = false;
        int gpu_id_ = 0;
        int gpu_mem_ = 4000;
        int cpu_math_library_num_threads_ = 4;
        bool use_mkldnn_ = false;

        std::string limit_type_ = "max";
        int limit_side_len_ = 960;

        double det_db_thresh_ = 0.3;
        double det_db_box_thresh_ = 0.5;
        double det_db_unclip_ratio_ = 2.0;
        std::string det_db_score_mode_ = "slow";
        bool use_dilation_ = false;

        bool visualize_ = true;
        bool use_tensorrt_ = false;
        std::string precision_ = "fp32";

        float mean_[3] = {0.485f * 255, 0.456f * 255, 0.406f * 255};
        float scale_[3] = {1 / 0.229f / 255, 1 / 0.224f / 255, 1 / 0.225f / 255};
        bool is_scale_ = true;

        // pre-process
        ResizeImgType0 resize_op_;
        // Normalize normalize_op_;  // Desperate for use ncnn:Mat::normalize instand
        // Permute permute_op_;

        // post-process
        DBPostProcessor post_processor_;
    };
} // namespace PaddleOCR
