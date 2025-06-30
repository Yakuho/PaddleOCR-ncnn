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
#include <include/preprocess_op.h>
#include <include/utility.h>
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <Net.h>

namespace PaddleOCR {
    class CRNNRecognizer {
    public:
        // Init CRNNRecognizer Configuration
        explicit CRNNRecognizer(const std::string &model_dir, const bool &use_gpu,
                                const int &gpu_id, const int &gpu_mem,
                                const int &cpu_math_library_num_threads,
                                const bool &use_mkldnn, const std::string &label_path,
                                const bool &use_tensorrt,
                                const std::string &precision,
                                const int &rec_batch_num, const int &rec_img_h,
                                const int &rec_img_w) noexcept;

        // Load Paddle inference model
        void LoadModel(const std::string &model_dir) noexcept;

        // Run predictor
        void Run(const std::vector<cv::Mat> &img_list,
                 std::vector<std::string> &rec_texts,
                 std::vector<float> &rec_text_scores,
                 std::vector<double> &times) noexcept;

    private:
        ncnn::Net predictor_;

        bool use_gpu_ = false;
        int gpu_id_ = 0;
        int gpu_mem_ = 4000;
        int cpu_math_library_num_threads_ = 4;
        bool use_mkldnn_ = false;

        std::vector<std::string> label_list_;

        float mean_[3] = {0.5f * 255, 0.5f * 255, 0.5f * 255};
        float scale_[3] = {1 / 0.5f / 255, 1 / 0.5f / 255, 1 / 0.5f / 255};
        bool is_scale_ = true;
        bool use_tensorrt_ = false;
        std::string precision_ = "fp32";
        int rec_batch_num_ = 6;
        int rec_img_h_ = 32;
        int rec_img_w_ = 320;
        std::vector<int> rec_image_shape_ = {3, rec_img_h_, rec_img_w_};

        // pre-process
        CrnnResizeImg resize_op_;
        // Normalize normalize_op_;
        // PermuteBatch permute_op_;
    }; // class CrnnRecognizer
} // namespace PaddleOCR
