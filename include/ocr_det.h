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
        explicit DBDetector(const std::string &model_dir, const bool &use_gpu,
                            const int &gpu_id, const int &gpu_mem,
                            const int &cpu_math_library_num_threads,
                            const bool &use_mkldnn, const std::string &limit_type,
                            const int &limit_side_len, const double &det_db_thresh,
                            const float &det_db_box_thresh,
                            const float &det_db_unclip_ratio,
                            const std::string &det_db_score_mode,
                            const bool &use_dilation, const bool &use_tensorrt,
                            const std::string &precision) noexcept {
            this->use_gpu_ = use_gpu;
            this->gpu_id_ = gpu_id;
            this->gpu_mem_ = gpu_mem;
            this->cpu_math_library_num_threads_ = cpu_math_library_num_threads;
            this->use_mkldnn_ = use_mkldnn;

            this->limit_type_ = limit_type;
            this->limit_side_len_ = limit_side_len;

            this->det_db_thresh_ = det_db_thresh;
            this->det_db_box_thresh_ = det_db_box_thresh;
            this->det_db_unclip_ratio_ = det_db_unclip_ratio;
            this->det_db_score_mode_ = det_db_score_mode;
            this->use_dilation_ = use_dilation;

            this->use_tensorrt_ = use_tensorrt;
            this->precision_ = precision;

            std::string yaml_file_path = model_dir + "/inference.yml";
            if (std::ifstream yaml_file(yaml_file_path); yaml_file.is_open()) {
                try {
                    // std::string model_name;
                    YAML::Node config = YAML::LoadFile(yaml_file_path);
                    // if (config["Global"] && config["Global"]["model_name"]) {
                    //     model_name = config["Global"]["model_name"].as<std::string>();
                    // }
                    // if (!model_name.empty() && model_name != "PP-OCRv5_mobile_det" &&
                    //     model_name != "PP-OCRv5_server_det") {
                    //     std::cerr << "Error: " << model_name << " is currently not supported."
                    //               << std::endl;
                    //     std::exit(EXIT_FAILURE);
                    //     }
                    if (config["PreProcess"]) {
                        const auto PreProcess = config["PreProcess"];
                        if (PreProcess["DetResizeForTest"] && PreProcess["DetResizeForTest"]["resize_long"])
                            this->limit_side_len_ = PreProcess["DetResizeForTest"]["resize_long"].as<int>();
                        if (PreProcess["NormalizeImage"]) {
                            if (PreProcess["NormalizeImage"]["mean"]) {
                                const auto mean = PreProcess["NormalizeImage"]["mean"];
                                for (std::size_t i = 0; i < 3; ++i)
                                    this->mean_[i] = mean[i].as<float>() * 255.0f;
                            }
                            if (PreProcess["NormalizeImage"]["std"]) {
                                const auto std = PreProcess["NormalizeImage"]["std"];
                                for (std::size_t i = 0; i < 3; ++i)
                                    this->scale_[i] = 1 / std[i].as<float>() / 255.0f;
                            }
                        }
                    }
                    if (config["PostProcess"]) {
                        if (config["PostProcess"]["thresh"])
                            this->det_db_thresh_ = config["PostProcess"]["thresh"].as<double>() * 255;
                        if (config["PostProcess"]["box_thresh"])
                            this->det_db_box_thresh_ = config["PostProcess"]["box_thresh"].as<float>();
                        if (config["PostProcess"]["unclip_ratio"])
                            this->det_db_unclip_ratio_ = config["PostProcess"]["unclip_ratio"].as<float>();
                    }
                } catch (const YAML::Exception &e) {
                    std::cerr << "Failed to load YAML file: " << e.what() << std::endl;
                }
            }

            LoadModel(model_dir);
        }

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
        float det_db_box_thresh_ = 0.5;
        float det_db_unclip_ratio_ = 2.0;
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
