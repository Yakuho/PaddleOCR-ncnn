# PaddleOCR-ncnn Inference

This project base from [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/tree/4602329be9432db4328f28a3e16a04a9eb8e823e/deploy/cpp_infer)
and Inference engine using Tencent [ncnn](https://github.com/Tencent/ncnn).

- **TODO:**

   - [x] PaddleOCR (det,cls,rec)
     - [x] PP-OCRv5
     - [x] PP-OCRv4
   - [ ] PaddleOCR-Struct
   - [ ] GPU Supported
   - [ ] Thread num Supported
   - [ ] Precision FP16、INT8、etc.
   - [ ] Memory Optimize

## 1. Prepare the Environment

### 1.1 Environment

- Linux, Windows, etc.

### 1.2 Third-party Dependencies

- [OpenCV](https://github.com/opencv/opencv)
- [ncnn](https://github.com/Tencent/ncnn)
- [yaml-cpp](https://github.com/jbeder/yaml-cpp)
- [glog](https://github.com/google/glog)
- [gflags](https://github.com/gflags/gflags)

### 2. Compile and Run the Demo

#### Build and Compile

- **Windows**

   If you wants to compile Windows version, maybe you need setting FLAG about xxx_deps_DIR before build 
   (make sure compiler can find these dependencies), It can set to cmake command, cmakelist file or Environment Path like:

  - Execute command: 

      ```shell
      -DOpenCV_DIR <path-to-OpenCV-cmake-root>/lib
      -Dncnn_DIR <path-to-ncnn-cmake-root>/lib/cmake/ncnn
      -Dglog_DIR <path-to-glog-cmake-root>/lib/cmake/glog
      -Dyaml-cpp_DIR <path-to-yaml-cpp-cmake-root>/lib/cmake/yaml-cpp
      -Dgflags_DIR <path-to-gflags-cmake-root>/gflags/lib/cmake/gflags
      ```

  - CMakeLists

      ```cmake
      set(OpenCV_DIR      <path-to-OpenCV-cmake-root>/lib)
      set(ncnn_DIR        <path-to-ncnn-cmake-root>/lib/cmake/ncnn)
      set(glog_DIR        <path-to-glog-cmake-root>/lib/cmake/glog)
      set(yaml-cpp_DIR    <path-to-yaml-cpp-cmake-root>/lib/cmake/yaml-cpp)
      set(gflags_DIR      <path-to-gflags-cmake-root>/gflags/lib/cmake/gflags)
      ```
    
- **Linux/Mac**

   Normally, Linux/Mac could install these dependencies by package manager like apt/yum/brew, and package manager will 
   auto set to global path like `/usr/lib`, so compiler can find these from global path.

   Linux can install these like: (ncnn maybe must build and install from source code)

   ```bash
   apt install libopencv-dev libgoogle-glog-dev libyaml-cpp-dev libgflags-dev
   ```
  
   or build and compile it from source code.

#### Tips:

For manage project conveniently, you can also link dependencies root to project root like:

- windows

   ```shell
   New-Item -ItemType Junction -Path "<path-to>\PaddleOCR-ncnn\3rdparty\yaml-cpp" -Target "<path-to>\yaml-cpp\build\install"
   New-Item -ItemType Junction -Path "<path-to>\PaddleOCR-ncnn\3rdparty\ncnn"     -Target "<path-to>\ncnn\build\install"
   ...
   ```

- linux

   ```bash
   ln -s "/path/to/yaml-cpp/build/install" "/path/to/PaddleOCR-ncnn/3rdparty/yaml-cpp"
   ln -s "/path/to/ncnn/build/install"     "/path/to/PaddleOCR-ncnn/3rdparty/ncnn"
   ...
   ```

#### Run Demo

1. **Prepare checkpoints:**

    ```
    PaddleOCR-ncnn/
    ├── checkpoints/
    │   ├── PP-LCNet_x0_25_textline_ori_infer/
    │   │   ├── inference.pdiparams
    │   │   ├── inference.json
    │   │   └── inference.yml
    │   ├── PP-OCRv5_mobile_det_infer/
    │   └── PP-OCRv5_mobile_rec_infer/
    ├── external-cmake/
    ├── include/
    ...
    ```
   
2. **paddle2onnx:**

    Document: [paddle2onnx](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/obtaining_onnx_models.html)
    
    Specially, paddlex args should set `--opset_version 11+`, and you can check FAQ maybe you will meet some trouble when converted.
    
    ```bash
    paddlex --paddle2onnx --paddle_model_dir <checkpoints>/PP-OCRv5_mobile_det_infer --onnx_model_dir <checkpoints>/PP-OCRv5_mobile_det_infer_onnx --opset_version 11
    ```

3. **onnx2ncnn:**

    Document: [onnx2ncnn](https://github.com/Tencent/ncnn/wiki/use-ncnn-with-pytorch-or-onnx)
    
    Specially, `pnnx` args `inputshape` and `inputshape2` can be found from `<checkpoints>/PP-OCRv5_mobile_det_infer_onnx/inference.yml`
    
    ```bash
    cd <checkpoints>/PP-OCRv5_mobile_det_infer_onnx/
    pnnx inference.onnx inputshape=[1,3,32,32] inputshape2=[1,3,736,736]
    ```

### 3. FAQ

1. Encountered the error `g++.exe .../extern_autolog-src/auto_log/autolog.h: In member function 'void AutoLogger::report()':
   .../extern_autolog-src/auto_log/autolog.h:66:27: error: 'accumulate' is not a member of 'std'; did you mean 'cv::accumulate'?
   66 |                   << std::accumulate(this->time_info_.begin(), this->time_info_.end(), 0);
   |                           ^~~~~~~~~~
   ...`, 
**change the file** `auto_log.h` from `<build_root>/third-party/extern_autolog-src/auto_log/autolog.h`

2. Encountered the error `[ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Failed to load model with error: /onnxruntime_src/onnxruntime/core/graph/model.cc:180 onnxruntime::Model::Model(onnx::ModelProto&&, const onnxruntime::PathString&, const onnxruntime::IOnnxRuntimeOpSchemaRegistryList*, const onnxruntime::logging::Logger&, const onnxruntime::ModelOptions&) Unsupported model IR version: 11, max supported IR version: 10
   `
**change Version** to Python site-package `onnx` to `onnx==1.7.0`