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

#include <include/ocr_rec.h>
#include <iostream>
#include <chrono>
#include <numeric>

namespace PaddleOCR {
    CRNNRecognizer::CRNNRecognizer(const std::string &model_dir, const bool &use_gpu,
                                   const int &gpu_id, const int &gpu_mem,
                                   const int &cpu_math_library_num_threads,
                                   const bool &use_mkldnn, const std::string &label_path,
                                   const bool &use_tensorrt,
                                   const std::string &precision,
                                   const int &rec_batch_num, const int &rec_img_h,
                                   const int &rec_img_w) noexcept {
        this->use_gpu_ = use_gpu;
        this->gpu_id_ = gpu_id;
        this->gpu_mem_ = gpu_mem;
        this->cpu_math_library_num_threads_ = cpu_math_library_num_threads;
        this->use_mkldnn_ = use_mkldnn;
        this->use_tensorrt_ = use_tensorrt;
        this->precision_ = precision;
        this->rec_batch_num_ = rec_batch_num;
        this->rec_img_h_ = rec_img_h;
        this->rec_img_w_ = rec_img_w;
        this->rec_image_shape_ = {3, rec_img_h, rec_img_w};

        std::string new_label_path = label_path;
        std::string yaml_file_path = model_dir + "/inference.yml";
        if (std::ifstream yaml_file(yaml_file_path); yaml_file.is_open()) {
            std::vector<std::string> rec_char_list;
            try {
                // std::string model_name;
                YAML::Node config = YAML::LoadFile(yaml_file_path);
                // if (config["Global"] && config["Global"]["model_name"]) {
                //     model_name = config["Global"]["model_name"].as<std::string>();
                // }
                // if (!model_name.empty() && model_name != "PP-OCRv5_mobile_rec" &&
                //     model_name != "PP-OCRv5_server_rec") {
                //     std::cerr << "Error: " << model_name << " is currently not supported."
                //             << std::endl;
                //     std::exit(EXIT_FAILURE);
                // }
                if (config["PreProcess"] && config["PreProcess"]["RecResizeImg"] &&
                    config["PreProcess"]["RecResizeImg"]["image_shape"]) {
                    const auto image_shape = config["PreProcess"]["RecResizeImg"]["image_shape"];
                    this->rec_image_shape_ = image_shape.as<std::vector<int> >();
                    this->rec_img_h_ = this->rec_image_shape_[1];
                    this->rec_img_w_ = this->rec_image_shape_[2];
                }
                if (config["PostProcess"] && config["PostProcess"]["character_dict"]) {
                    rec_char_list = config["PostProcess"]["character_dict"]
                            .as<std::vector<std::string> >();
                }
            } catch (const YAML::Exception &e) {
                std::cerr << "Failed to load YAML file: " << e.what() << std::endl;
            }
            if (label_path == "../../ppocr/utils/ppocr_keys_v1.txt" &&
                !rec_char_list.empty()) {
                std::string new_rec_char_dict_path = model_dir + "/ppocr_keys_v1.txt";
                if (std::ofstream new_file(new_rec_char_dict_path); new_file.is_open()) {
                    for (const auto &character: rec_char_list) {
                        new_file << character << '\n';
                    }
                    new_label_path = new_rec_char_dict_path;
                }
            }
        }

        this->label_list_ = Utility::ReadDict(new_label_path);
        this->label_list_.emplace(this->label_list_.begin(), "#"); // blank char for ctc
        this->label_list_.emplace_back(" ");

        LoadModel(model_dir);
    }

    void CRNNRecognizer::LoadModel(const std::string &model_dir) noexcept {
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
        // std::cout << "In PP-OCRv3, default rec_img_h is 48,"
        //           << "if you use other model, you should set the param rec_img_h=32"
        //           << std::endl;
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
        //         if (!Utility::PathExists("./trt_rec_shape.txt")) {
        //             config.CollectShapeRangeInfo("./trt_rec_shape.txt");
        //         } else {
        //             config.EnableTunedTensorRtDynamicShape("./trt_rec_shape.txt", true);
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
        //
        // // get pass_builder object
        // auto pass_builder = config.pass_builder();
        // // delete "matmul_transpose_reshape_fuse_pass"
        // pass_builder->DeletePass("matmul_transpose_reshape_fuse_pass");
        // config.SwitchUseFeedFetchOps(false);
        // // true for multiple input
        // config.SwitchSpecifyInputNames(true);
        //
        // config.SwitchIrOptim(true);
        //
        // config.EnableMemoryOptim();
        // //   config.DisableGlogInfo();

        // const std::string model_file_path = model_dir + "/inference.bin";
        // const std::string param_file_path = model_dir + "/inference.param";

        this->predictor_.load_param(param_file_path.c_str());
        if (const int ret = this->predictor_.load_model(model_file_path.c_str()); ret != 0) {
            std::cerr << "[ERROR] load cls model failed" << std::endl;
            exit(1);
        }
    }

    void CRNNRecognizer::Run(const std::vector<cv::Mat> &img_list,
                             std::vector<std::string> &rec_texts,
                             std::vector<float> &rec_text_scores,
                             std::vector<double> &times) noexcept {
        std::chrono::duration<float> preprocess_diff =
                std::chrono::duration<float>::zero();
        std::chrono::duration<float> inference_diff =
                std::chrono::duration<float>::zero();
        std::chrono::duration<float> postprocess_diff =
                std::chrono::duration<float>::zero();

        size_t img_num = img_list.size();
        std::vector<float> width_list;
        for (size_t i = 0; i < img_num; ++i)
            width_list.emplace_back(static_cast<float>(img_list[i].cols) /
                                    static_cast<float>(img_list[i].rows));
        std::vector<size_t> indices = std::move(Utility::argsort(width_list));

        const int imgH = this->rec_image_shape_[1];
        const int imgW = this->rec_image_shape_[2];
        const float max_wh_ratio_init = static_cast<float>(imgW) / static_cast<float>(imgH);
        for (size_t beg_img_no = 0; beg_img_no < img_num;
             beg_img_no += this->rec_batch_num_) {
            auto preprocess_start = std::chrono::steady_clock::now();
            size_t end_img_no = std::min(img_num, beg_img_no + this->rec_batch_num_);
            int batch_num = static_cast<int>(end_img_no - beg_img_no);
            float max_wh_ratio = max_wh_ratio_init;
            for (size_t ino = beg_img_no; ino < end_img_no; ++ino)
                max_wh_ratio = std::max(max_wh_ratio, width_list[indices[ino]]);

            int batch_width = imgW;
            std::vector<ncnn::Mat> norm_img_batch(batch_num);
            for (size_t ino = beg_img_no; ino < end_img_no; ++ino) {
                cv::Mat srcimg;
                img_list[indices[ino]].copyTo(srcimg);
                cv::Mat resize_img;
                this->resize_op_.Run(srcimg, resize_img, max_wh_ratio,
                                     this->use_tensorrt_, this->rec_image_shape_);
                ncnn::Mat input = ncnn::Mat::from_pixels(
                    resize_img.data, ncnn::Mat::PIXEL_BGR, resize_img.cols, resize_img.rows);
                input.substract_mean_normalize(this->mean_, this->scale_);
                batch_width = std::max(resize_img.cols, batch_width);
                norm_img_batch[ino] = input;
            }
            auto preprocess_end = std::chrono::steady_clock::now();
            preprocess_diff += preprocess_end - preprocess_start;

            // Inference.
            auto inference_start = std::chrono::steady_clock::now();
            constexpr int model_dspr = 8; // TODO: Difference model maybe has difference downsample rate !!!
            const int predict_width = batch_width / model_dspr;
            std::vector<int> predict_shape = {batch_num, predict_width, static_cast<int>(this->label_list_.size())};

            const int predict_num = predict_shape[1] * predict_shape[2];
            std::vector<float> predict_batch(batch_num * predict_num);
            for (int ino = static_cast<int>(beg_img_no); ino < end_img_no; ++ino) {
                ncnn::Extractor extractor = this->predictor_.create_extractor();
                extractor.input(predictor_.input_names()[0], norm_img_batch[ino]);
                ncnn::Mat output;
                extractor.extract(predictor_.output_names()[0], output);
                const int out_num = output.h * output.w * output.c;
                ncnn::Mat out_data = output.reshape(out_num);
                std::memcpy(predict_batch.data() + ino * predict_num, &out_data[0],
                    out_num * sizeof(float));
            }
            auto inference_end = std::chrono::steady_clock::now();
            inference_diff += inference_end - inference_start;

            // ctc decode
            auto postprocess_start = std::chrono::steady_clock::now();
            for (int m = 0; m < predict_shape[0]; ++m) {
                std::string str_res;
                int argmax_idx;
                int last_index = 0;
                float score = 0.f;
                float count = 0.f;
                float max_value;

                for (int n = 0; n < predict_shape[1]; ++n) {
                    argmax_idx = static_cast<int>(
                        Utility::argmax(
                            &predict_batch[(m * predict_shape[1] + n) * predict_shape[2]],
                            &predict_batch[(m * predict_shape[1] + n + 1) * predict_shape[2]])
                    );
                    max_value = predict_batch[(m * predict_shape[1] + n) * predict_shape[2] + argmax_idx];

                    if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index))) {
                        score += max_value;
                        count += 1.0;
                        str_res += label_list_[argmax_idx];
                    }
                    last_index = argmax_idx;
                }
                score /= count;
                if (std::isnan(score)) {
                    continue;
                }
                rec_texts[indices[beg_img_no + m]] = std::move(str_res);
                rec_text_scores[indices[beg_img_no + m]] = score;
            }
            auto postprocess_end = std::chrono::steady_clock::now();
            postprocess_diff += postprocess_end - postprocess_start;
        }
        times.emplace_back(preprocess_diff.count() * 1000);
        times.emplace_back(inference_diff.count() * 1000);
        times.emplace_back(postprocess_diff.count() * 1000);
    }
} // namespace PaddleOCR
