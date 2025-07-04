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

#include <include/args.h>
#include <include/ocr_cls.h>
#include <include/ocr_det.h>
#include <include/ocr_rec.h>
#include <include/paddleocr.h>
#include <auto_log/autolog.h>

namespace PaddleOCR {
    struct PPOCR::PPOCR_PRIVATE {
        std::unique_ptr<DBDetector> detector_;
        std::unique_ptr<Classifier> classifier_;
        std::unique_ptr<CRNNRecognizer> recognizer_;
    };

    PPOCR::PPOCR() noexcept : pri_(new PPOCR_PRIVATE) {
        if (FLAGS_det) {
            this->pri_->detector_ = std::make_unique<DBDetector>(
                FLAGS_det_model_dir, FLAGS_use_gpu, FLAGS_gpu_id, FLAGS_gpu_mem,
                FLAGS_cpu_threads, FLAGS_enable_mkldnn, FLAGS_limit_type,
                FLAGS_limit_side_len, FLAGS_det_db_thresh, FLAGS_det_db_box_thresh,
                FLAGS_det_db_unclip_ratio, FLAGS_det_db_score_mode, FLAGS_use_dilation,
                FLAGS_use_tensorrt, FLAGS_precision);
        }
        if (FLAGS_cls && FLAGS_use_angle_cls) {
            this->pri_->classifier_ = std::make_unique<Classifier>(
                FLAGS_cls_model_dir, FLAGS_use_gpu, FLAGS_gpu_id, FLAGS_gpu_mem,
                FLAGS_cpu_threads, FLAGS_enable_mkldnn, FLAGS_cls_thresh,
                FLAGS_use_tensorrt, FLAGS_precision, FLAGS_cls_batch_num);
        }
        if (FLAGS_rec) {
            this->pri_->recognizer_ = std::make_unique<CRNNRecognizer>(
                FLAGS_rec_model_dir, FLAGS_use_gpu, FLAGS_gpu_id, FLAGS_gpu_mem,
                FLAGS_cpu_threads, FLAGS_enable_mkldnn, FLAGS_rec_char_dict_path,
                FLAGS_use_tensorrt, FLAGS_precision, FLAGS_rec_batch_num,
                FLAGS_rec_img_h, FLAGS_rec_img_w);
        }
    }

    PPOCR::~PPOCR() { delete this->pri_; }

    std::vector<std::vector<OCRPredictResult> >
    PPOCR::ocr(const std::vector<cv::Mat> &img_list,
               const bool det, const bool rec, const bool cls) noexcept {
        std::vector<std::vector<OCRPredictResult> > ocr_results;

        if (!det) {
            std::vector<OCRPredictResult> ocr_result(img_list.size());
            if (cls && this->pri_->classifier_) {
                this->cls(img_list, ocr_result);
                for (size_t i = 0; i < img_list.size(); ++i) {
                    if (ocr_result[i].cls_label % 2 == 1 &&
                        ocr_result[i].cls_score > this->pri_->classifier_->cls_thresh) {
                        cv::rotate(img_list[i], img_list[i], 1);
                    }
                }
            }
            if (rec) {
                this->rec(img_list, ocr_result);
            }
            for (auto &i: ocr_result) {
                ocr_results.emplace_back(1, std::move(i));
            }
        } else {
            for (const auto & img: img_list) {
                std::vector<OCRPredictResult> ocr_result =
                        this->ocr(img, true, rec, cls);
                ocr_results.emplace_back(std::move(ocr_result));
            }
        }
        return ocr_results;
    }

    std::vector<OCRPredictResult>
    PPOCR::ocr(const cv::Mat &img,
               const bool det, const bool rec, const bool cls) noexcept {
        std::vector<OCRPredictResult> ocr_result;
        // det
        this->det(img, ocr_result);
        // crop image
        std::vector<cv::Mat> img_list;
        for (auto & j: ocr_result) {
            cv::Mat crop_img = Utility::GetRotateCropImage(img, j.box);
            img_list.emplace_back(std::move(crop_img));
        }
        // cls
        if (cls && this->pri_->classifier_) {
            this->cls(img_list, ocr_result);
            for (size_t i = 0; i < img_list.size(); ++i) {
                if (ocr_result[i].cls_label % 2 == 1 &&
                    ocr_result[i].cls_score > this->pri_->classifier_->cls_thresh) {
                    cv::rotate(img_list[i], img_list[i], 1);
                }
            }
        }
        // rec
        if (rec) {
            this->rec(img_list, ocr_result);
        }
        return ocr_result;
    }

    void PPOCR::det(const cv::Mat &img,
                    std::vector<OCRPredictResult> &ocr_results) noexcept {
        std::vector<std::vector<std::vector<int> > > boxes;
        std::vector<double> det_times;

        this->pri_->detector_->Run(img, boxes, det_times);
        for (auto & box: boxes) {
            OCRPredictResult res;
            res.box = std::move(box);
            ocr_results.emplace_back(std::move(res));
        }
        // sort boxes from top to bottom, from left to right
        Utility::sort_boxes(ocr_results);
        this->time_info_det[0] += det_times[0];
        this->time_info_det[1] += det_times[1];
        this->time_info_det[2] += det_times[2];
    }

    void PPOCR::rec(const std::vector<cv::Mat> &img_list,
                    std::vector<OCRPredictResult> &ocr_results) noexcept {
        std::vector<std::string> rec_texts(img_list.size(), std::string());
        std::vector<float> rec_text_scores(img_list.size(), 0);
        std::vector<double> rec_times;

        this->pri_->recognizer_->Run(img_list, rec_texts, rec_text_scores, rec_times);
        // output rec results
        for (size_t i = 0; i < rec_texts.size(); ++i) {
            ocr_results[i].text = std::move(rec_texts[i]);
            ocr_results[i].score = rec_text_scores[i];
        }
        this->time_info_rec[0] += rec_times[0];
        this->time_info_rec[1] += rec_times[1];
        this->time_info_rec[2] += rec_times[2];
    }

    void PPOCR::cls(const std::vector<cv::Mat> &img_list,
                    std::vector<OCRPredictResult> &ocr_results) noexcept {
        std::vector<int> cls_labels(img_list.size(), 0);
        std::vector<float> cls_scores(img_list.size(), 0);
        std::vector<double> cls_times;

        this->pri_->classifier_->Run(img_list, cls_labels, cls_scores, cls_times);
        // output cls results
        for (size_t i = 0; i < cls_labels.size(); ++i) {
            ocr_results[i].cls_label = cls_labels[i];
            ocr_results[i].cls_score = cls_scores[i];
        }
        this->time_info_cls[0] += cls_times[0];
        this->time_info_cls[1] += cls_times[1];
        this->time_info_cls[2] += cls_times[2];
    }

    void PPOCR::reset_timer() noexcept {
        this->time_info_det = {0, 0, 0};
        this->time_info_rec = {0, 0, 0};
        this->time_info_cls = {0, 0, 0};
    }

    void PPOCR::benchmark_log(int img_num) const noexcept {
        if (this->time_info_det[0] + this->time_info_det[1] + this->time_info_det[2] > 0) {
            AutoLogger autolog_det("ocr_det", FLAGS_use_gpu, FLAGS_use_tensorrt,
                                   FLAGS_enable_mkldnn, FLAGS_cpu_threads,
                                   1, "dynamic", FLAGS_precision,
                                   this->time_info_det, img_num);
            autolog_det.report();
        }
        if (this->time_info_rec[0] + this->time_info_rec[1] + this->time_info_rec[2] > 0) {
            AutoLogger autolog_rec("ocr_rec", FLAGS_use_gpu, FLAGS_use_tensorrt,
                                   FLAGS_enable_mkldnn, FLAGS_cpu_threads,
                                   FLAGS_rec_batch_num, "dynamic", FLAGS_precision,
                                   this->time_info_rec, img_num);
            autolog_rec.report();
        }
        if (this->time_info_cls[0] + this->time_info_cls[1] + this->time_info_cls[2] > 0) {
            AutoLogger autolog_cls("ocr_cls", FLAGS_use_gpu, FLAGS_use_tensorrt,
                                   FLAGS_enable_mkldnn, FLAGS_cpu_threads,
                                   FLAGS_cls_batch_num, "dynamic", FLAGS_precision,
                                   this->time_info_cls, img_num);
            autolog_cls.report();
        }
    }
} // namespace PaddleOCR
