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

#include <include/ocr_cls.h>
#include <iostream>
#include <chrono>
#include <numeric>

namespace PaddleOCR {
    Classifier::Classifier(const std::string &model_dir, const bool &use_gpu,
                           const int &gpu_id, const int &gpu_mem,
                           const int &cpu_math_library_num_threads,
                           const bool &use_mkldnn, const double &cls_thresh,
                           const bool &use_tensorrt, const std::string &precision,
                           const int &cls_batch_num) noexcept {
        this->use_gpu_ = use_gpu;
        this->gpu_id_ = gpu_id;
        this->gpu_mem_ = gpu_mem;
        this->cpu_math_library_num_threads_ = cpu_math_library_num_threads;
        this->use_mkldnn_ = use_mkldnn;

        this->cls_thresh = cls_thresh;
        this->use_tensorrt_ = use_tensorrt;
        this->precision_ = precision;
        this->cls_batch_num_ = cls_batch_num;

        std::string yaml_file_path = model_dir + "/inference.yml";
        if (std::ifstream yaml_file(yaml_file_path); yaml_file.is_open()) {
            try {
                std::string model_name;
                YAML::Node config = YAML::LoadFile(yaml_file_path);
                // if (config["Global"] && config["Global"]["model_name"]) {
                //     model_name = config["Global"]["model_name"].as<std::string>();
                // }
                // if (!model_name.empty() &&
                //     model_name != "PP-LCNet_x0_25_textline_ori" &&
                //     model_name != "PP-LCNet_x1_0_textline_ori") {
                //     std::cerr << "Error: " << model_name << " is currently not supported."
                //             << std::endl;
                //     std::exit(EXIT_FAILURE);
                // }
                if (config["PreProcess"] && config["PreProcess"]["transform_ops"]) {
                    const auto transform_ops = config["PreProcess"]["transform_ops"];
                    if (transform_ops["ResizeImage"] && transform_ops["ResizeImage"]["size"]) {
                        const auto size = transform_ops["ResizeImage"]["size"];
                        this->cls_image_shape_[2] = size[0].as<int>();
                        this->cls_image_shape_[1] = size[1].as<int>();
                    }
                    if (transform_ops["NormalizeImage"] && transform_ops["NormalizeImage"]["mean"]) {
                        const auto mean = transform_ops["NormalizeImage"]["mean"];
                        for (int i = 0; i < 3; i++)
                            this->mean_[i] = mean[i].as<float>() * 255;
                    }
                    if (transform_ops["NormalizeImage"] && transform_ops["NormalizeImage"]["scale"]) {
                        const auto scale = transform_ops["NormalizeImage"]["scale"];
                        for (int i = 0; i < 3; i++)
                            this->scale_[i] = 1 / scale[i].as<float>() / 255;
                    }
                }
            } catch (const YAML::Exception &e) {
                std::cerr << "Failed to load YAML file: " << e.what() << std::endl;
            }
        }

        LoadModel(model_dir);
    }

    void Classifier::LoadModel(const std::string &model_dir) noexcept {
        // TODO: Setting some FLAGS like vulkan, etc. instand
        // paddle_infer::Config config;
        std::string model_file_path, param_file_path;
        std::vector<std::pair<std::string, std::string> > model_variants = {
            {"/inference.ncnn.bin", "/inference.ncnn.param"},
            {"/model.ncnn.bin", "/model.ncnn.param"},
            {"/inference.bin", "/inference.param"},
            {"/model.bin", "/model.param"}
        };
        for (const auto& [model_path, param_path]: model_variants) {
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
        //         config.EnableTensorRtEngine(1 << 20, 10, 3, precision, false, false);
        //         if (!Utility::PathExists("./trt_cls_shape.txt")) {
        //             config.CollectShapeRangeInfo("./trt_cls_shape.txt");
        //         } else {
        //             config.EnableTunedTensorRtDynamicShape("./trt_cls_shape.txt", true);
        //         }
        //     }
        // } else {
        //     config.DisableGpu();
        //     if (this->use_mkldnn_) {
        //         config.EnableMKLDNN();
        //     } else {
        //         config.DisableMKLDNN();
        //     }
        //     config.SetCpuMathLibraryNumThreads(this->cpu_math_library_num_threads_);
        // }
        //
        // // false for zero copy tensor
        // config.SwitchUseFeedFetchOps(false);
        // // true for multiple input
        // config.SwitchSpecifyInputNames(true);
        //
        // config.SwitchIrOptim(true);
        //
        // config.EnableMemoryOptim();
        // config.DisableGlogInfo();

        this->predictor_.load_param(param_file_path.c_str());
        if (const int ret = this->predictor_.load_model(model_file_path.c_str()); ret != 0) {
            std::cerr << "[ERROR] load cls model failed" << std::endl;
            exit(1);
        }
    }

    void Classifier::Run(const std::vector<cv::Mat> &img_list,
                         std::vector<int> &cls_labels,
                         std::vector<float> &cls_scores,
                         std::vector<double> &times) noexcept {
        std::chrono::duration<float> preprocess_diff =
                std::chrono::duration<float>::zero();
        std::chrono::duration<float> inference_diff =
                std::chrono::duration<float>::zero();
        std::chrono::duration<float> postprocess_diff =
                std::chrono::duration<float>::zero();

        int img_num = static_cast<int>(img_list.size());
        for (int beg_img_no = 0; beg_img_no < img_num; beg_img_no += this->cls_batch_num_) {
            auto preprocess_start = std::chrono::steady_clock::now();
            int end_img_no = std::min(img_num, beg_img_no + this->cls_batch_num_);
            int batch_num = end_img_no - beg_img_no;

            // preprocess
            std::vector<ncnn::Mat> norm_img_batch(batch_num);
            for (int ino = beg_img_no; ino < end_img_no; ++ino) {
                cv::Mat srcimg;
                img_list[ino].copyTo(srcimg);
                cv::Mat resize_img;
                this->resize_op_.Run(srcimg, resize_img, this->use_tensorrt_,
                                     this->cls_image_shape_);
                if (resize_img.cols < this->cls_image_shape_[2]) {
                    cv::copyMakeBorder(resize_img, resize_img, 0, 0, 0,
                                       this->cls_image_shape_[2] - resize_img.cols,
                                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
                }
                ncnn::Mat input = ncnn::Mat::from_pixels(
                    resize_img.data, ncnn::Mat::PIXEL_BGR, resize_img.cols, resize_img.rows);
                input.substract_mean_normalize(this->mean_, this->scale_);
                norm_img_batch[ino] = input;
            }
            auto preprocess_end = std::chrono::steady_clock::now();
            preprocess_diff += preprocess_end - preprocess_start;

            // inference.
            auto inference_start = std::chrono::steady_clock::now();
            constexpr int cls_num = 2;  // TODO: maybe new virsion cls model num not equal 2 !!!
            const std::vector<int> predict_shape = {batch_num, cls_num};
            std::vector<float> predict_batch(batch_num * cls_num);
            for (int ino = beg_img_no; ino < end_img_no; ++ino) {
                ncnn::Extractor extractor = this->predictor_.create_extractor();
                extractor.input(predictor_.input_names()[0], norm_img_batch[ino]);
                ncnn::Mat output;
                extractor.extract(predictor_.output_names()[0], output);
                const int out_num = output.h * output.w * output.c;
                ncnn::Mat out_data = output.reshape(out_num);
                std::memcpy(predict_batch.data() + ino * cls_num, &out_data[0], cls_num * sizeof(float));
            }
            auto inference_end = std::chrono::steady_clock::now();
            inference_diff += inference_end - inference_start;

            // postprocess
            auto postprocess_start = std::chrono::steady_clock::now();
            for (int batch_idx = 0; batch_idx < predict_shape[0]; ++batch_idx) {
                const float* beg_add = &predict_batch[(batch_idx)     * predict_shape[1]];
                const float* end_add = &predict_batch[(batch_idx + 1) * predict_shape[1]];
                const int argmax_idx = static_cast<int>(Utility::argmax(beg_add, end_add));
                float score = predict_batch[batch_idx * predict_shape[1] + argmax_idx];
                cls_labels[beg_img_no + batch_idx] = argmax_idx;
                cls_scores[beg_img_no + batch_idx] = score;
            }
            auto postprocess_end = std::chrono::steady_clock::now();
            postprocess_diff += postprocess_end - postprocess_start;
        }
        times.emplace_back(preprocess_diff.count() * 1000);
        times.emplace_back(inference_diff.count() * 1000);
        times.emplace_back(postprocess_diff.count() * 1000);
    }
} // namespace PaddleOCR
