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

#include <include/ocr_det.h>
#include <iostream>

namespace PaddleOCR {
    DBDetector::DBDetector(const std::string &model_dir, const bool &use_gpu,
                           const int &gpu_id, const int &gpu_mem,
                           const int &cpu_math_library_num_threads,
                           const bool &use_mkldnn, const std::string &limit_type,
                           const int &limit_side_len, const double &det_db_thresh,
                           const double &det_db_box_thresh,
                           const double &det_db_unclip_ratio,
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

    void DBDetector::LoadModel(const std::string &model_dir) noexcept {
        // TODO: Setting some FLAGS like vulkan, etc. instand
        // paddle_infer::Config config;
        std::string model_file_path, param_file_path;
        std::vector<std::pair<std::string, std::string> > model_variants = {
            {"/inference.ncnn.bin", "/inference.ncnn.param"},
            {"/model.ncnn.bin", "/model.ncnn.param"},
            {"/inference.bin", "/inference.param"},
            {"/model.bin", "/model.param"}
        };
        for (const auto &[model_path, param_path]: model_variants) {
            if (Utility::PathExists(model_dir + model_path) &&
                Utility::PathExists(model_dir + param_path)) {
                model_file_path = model_dir + model_path;
                param_file_path = model_dir + param_path;
                break;
            }
        }
        if (model_file_path.empty() || param_file_path.empty()) {
            std::cerr << "[ERROR] No valid model file found in " << model_dir << std::endl;
            exit(1);
        }
        // config.SetModel(model_file_path, param_file_path);
        // if (this->use_gpu_) {
        //     config.EnableUseGpu(this->gpu_mem_, this->gpu_id_);
        //     if (this->use_tensorrt_) {
        //         auto precision = paddle_infer::Config::Precision::kFloat32;
        //         if (this->precision_ == "fp16") {
        //             precision = paddle_infer::Config::Precision::kHalf;
        //         }
        //         if (this->precision_ == "int8") {
        //             precision = paddle_infer::Config::Precision::kInt8;
        //         }
        //         config.EnableTensorRtEngine(1 << 30, 1, 20, precision, false, false);
        //         if (!Utility::PathExists("./trt_det_shape.txt")) {
        //             config.CollectShapeRangeInfo("./trt_det_shape.txt");
        //         } else {
        //             config.EnableTunedTensorRtDynamicShape("./trt_det_shape.txt", true);
        //         }
        //     }
        // } else {
        //     config.DisableGpu();
        //     if (this->use_mkldnn_) {
        //         config.EnableMKLDNN();
        //         // cache 10 different shapes for mkldnn to avoid memory leak
        //         config.SetMkldnnCacheCapacity(10);
        //     } else {
        //         config.DisableMKLDNN();
        //     }
        //     config.SetCpuMathLibraryNumThreads(this->cpu_math_library_num_threads_);
        // }
        // // use zero_copy_run as default
        // config.SwitchUseFeedFetchOps(false);
        // // true for multiple input
        // config.SwitchSpecifyInputNames(true);
        //
        // config.SwitchIrOptim(true);
        //
        // config.EnableMemoryOptim();
        // // config.DisableGlogInfo();

        this->predictor_.load_param(param_file_path.c_str());
        if (const int ret = this->predictor_.load_model(model_file_path.c_str()); ret != 0) {
            std::cerr << "[ERROR] load det model failed" << std::endl;
            exit(1);
        }
    }

    void DBDetector::Run(const cv::Mat &img,
                         std::vector<std::vector<std::vector<int> > > &boxes,
                         std::vector<double> &times) noexcept {
        std::chrono::duration<float> preprocess_diff =
                std::chrono::duration<float>::zero();
        std::chrono::duration<float> inference_diff =
                std::chrono::duration<float>::zero();
        std::chrono::duration<float> postprocess_diff =
                std::chrono::duration<float>::zero();

        float ratio_h{};
        float ratio_w{};

        cv::Mat srcimg;
        cv::Mat resize_img;
        img.copyTo(srcimg);

        // Preprocess.
        auto preprocess_start = std::chrono::steady_clock::now();
        this->resize_op_.Run(img, resize_img, this->limit_type_,
                             this->limit_side_len_, ratio_h, ratio_w,
                             this->use_tensorrt_);
        ncnn::Mat input = ncnn::Mat::from_pixels(
            resize_img.data, ncnn::Mat::PIXEL_BGR, resize_img.cols, resize_img.rows);
        input.substract_mean_normalize(mean_, scale_);
        auto preprocess_end = std::chrono::steady_clock::now();
        preprocess_diff += preprocess_end - preprocess_start;

        // Inference.
        auto inference_start = std::chrono::steady_clock::now();
        ncnn::Extractor extractor = this->predictor_.create_extractor();
        extractor.input(predictor_.input_names()[0], input);
        ncnn::Mat output;
        extractor.extract(predictor_.output_names()[0], output);
        const int out_num = output.h * output.w * output.c;
        ncnn::Mat out_data = output.reshape(out_num);
        auto inference_end = std::chrono::steady_clock::now();
        inference_diff += inference_end - inference_start;

        // Postprocess.
        auto postprocess_start = std::chrono::steady_clock::now();
        const int n2 = output.h;  // height
        const int n3 = output.w;  // width
        const int n = n2 * n3;

        std::vector<float> pred(n, 0.0);
        std::vector<unsigned char> cbuf(n, ' ');

        for (int i = 0; i < n; ++i) {
            pred[i] = out_data[i];
            cbuf[i] = static_cast<unsigned char>(out_data[i] * 255);
        }

        cv::Mat cbuf_map(n2, n3, CV_8UC1, (unsigned char *)cbuf.data());
        cv::Mat pred_map(n2, n3, CV_32F, (float *)pred.data());

        cv::Mat bit_map;
        cv::threshold(cbuf_map, bit_map, this->det_db_thresh_, 255, cv::THRESH_BINARY);
        if (this->use_dilation_) {
            cv::Mat dila_ele = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
            cv::dilate(bit_map, bit_map, dila_ele);
        }

        boxes = std::move(post_processor_.BoxesFromBitmap(
            pred_map, bit_map, this->det_db_box_thresh_, this->det_db_unclip_ratio_,
            this->det_db_score_mode_));

        post_processor_.FilterTagDetRes(boxes, ratio_h, ratio_w, srcimg);
        auto postprocess_end = std::chrono::steady_clock::now();
        postprocess_diff += postprocess_end - postprocess_start;

        times.emplace_back(preprocess_diff.count() * 1000);
        times.emplace_back(inference_diff.count() * 1000);
        times.emplace_back(postprocess_diff.count() * 1000);
    }
} // namespace PaddleOCR
