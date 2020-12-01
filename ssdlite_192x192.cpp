// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "net.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <vector>
#include <string>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else // _WIN32
#include <sys/time.h>
#endif // _WIN32


#ifdef _WIN32
// 单位ms
double get_current_time()
{
    LARGE_INTEGER freq;
    LARGE_INTEGER pc;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&pc);

    return pc.QuadPart * 1000.0 / freq.QuadPart;
}
#else  // _WIN32
// 单位ms
double get_current_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}
#endif // _WIN32

class Noop : public ncnn::Layer
{
};
DEFINE_LAYER_CREATOR(Noop)

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static int init(const std::string &model_path, ncnn::Net &model)
{
    assert(!model_path.empty());

    model.opt.num_threads = 4; // You need to compile with libgomp for multi thread support
    model.opt.use_vulkan_compute = true; // You need to compile with libvulkan for gpu support
    model.register_custom_layer("Silence", Noop_layer_creator);

    std::string param_file = "mobilenetv2_ssdlite_voc.param";
    std::string model_file = "mobilenetv2_ssdlite_voc.bin";

    if(model_path.back() == '/' || model_path.back() == '\\')
    {
        param_file = model_path + param_file;
        model_file = model_path + model_file;
    }
    else
    {
        param_file = model_path + "/" + param_file;
        model_file = model_path + "/" + model_file;
    }

    model.load_param(param_file.c_str());
    model.load_model(model_file.c_str());
    return 0;
}

static int detect(ncnn::Net &model, const cv::Mat& bgr, std::vector<Object>& objects)
{
    const int target_size = 192;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, target_size, target_size);

    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = model.create_extractor();

    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("detection_out", out);

    //     printf("%d %d %d\n", out.w, out.h, out.c);
    objects.clear();
    for (int i = 0; i < out.h; i++)
    {
        const float* values = out.row(i);

        Object object;
        object.label = values[0];
        object.prob = values[1];
        object.rect.x = values[2] * img_w;
        object.rect.y = values[3] * img_h;
        object.rect.width = values[4] * img_w - object.rect.x;
        object.rect.height = values[5] * img_h - object.rect.y;

        objects.push_back(object);
    }
    return 0;
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {"background",
                                        "aeroplane", "bicycle", "bird", "boat",
                                        "bottle", "bus", "car", "cat", "chair",
                                        "cow", "diningtable", "dog", "horse",
                                        "motorbike", "person", "pottedplant",
                                        "sheep", "sofa", "train", "tvmonitor"
                                       };

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    // cv::imshow("image", image);
    // cv::waitKey(0);
    cv::imwrite("result.jpg", image);
}

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s [modelpath] [imagepath]\n", argv[0]);
        return -1;
    }

    // std::string model_path = argv[1];
    const char* model_path = argv[1];
    const char* image_path = argv[2];

    cv::Mat m = cv::imread(image_path, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", image_path);
        return -1;
    }

    ncnn::Net model;
    double start = get_current_time();
    init(model_path, model);
    double end = get_current_time();
    double time = end - start;
    fprintf(stderr, "time of init is %.2f ms\n", time);

    std::vector<Object> objects;

    printf("start evaluate speed...\n");
    int num = 1;
    start = get_current_time();
    for(int i = 0; i < num; ++i)
    {
        detect(model, m, objects);
    }
    end = get_current_time();
    time = (end - start) / num;
    fprintf(stderr, "time of detect is %.2f ms\n", time);
    
    draw_objects(m, objects);

    return 0;
}
