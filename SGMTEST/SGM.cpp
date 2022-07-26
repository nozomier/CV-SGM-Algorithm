#include <stdio.h>
#include<stdlib.h>
#include <tchar.h>
#include "SGM.h"
//#include "filter.h"
#include<iostream>
#include<opencv2/opencv.hpp>
#include <opencv2\imgproc\types_c.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include<algorithm>
using namespace cv;
using namespace std;
SemiGlobalMatching::SemiGlobalMatching()
{
}



SemiGlobalMatching::~SemiGlobalMatching()
{
    if (census_left_ != nullptr) {
        delete[] census_left_;
        census_left_ = nullptr;
    }
    if (census_right_ != nullptr) {
        delete[] census_right_;
        census_right_ = nullptr;
    }
    if (cost_init_ != nullptr) {
        delete[] cost_init_;
        cost_init_ = nullptr;
    }
    if (cost_aggr_ != nullptr) {
        delete[] cost_aggr_;
        cost_aggr_ = nullptr;
    }
    if (disp_left_ != nullptr) {
        delete[] disp_left_;
        disp_left_ = nullptr;
    }
    is_initialized_ = false;
}

bool SemiGlobalMatching::Initialize(const sint32& width, const sint32& height, const SGMOption& option)
{

    // 影像尺寸
    width_ = width;
    height_ = height;
    // SGM参数
    option_ = option;

    if (width == 0 || height == 0) {
        return false;
    }

    // 开辟内存空间

    // census值（左右影像）
    census_left_ = new uint32[width * height]();
    census_right_ = new uint32[width * height]();

    // 匹配代价（初始/聚合）
   const sint32 disp_range = option.max_disparity - option.min_disparity;
    if (disp_range <= 0) {
        return false;
    }
    //cost_init_ = new uint8[width * height * disp_range]();
    //cost_aggr_ = new uint16[width * height * disp_range]();

    const sint32 size = width * height * disp_range;
    cost_init_ = new uint8[size]();
    cost_aggr_ = new uint16[size]();
    cost_aggr_1_ = new uint8[size]();
    cost_aggr_2_ = new uint8[size]();
    cost_aggr_3_ = new uint8[size]();
    cost_aggr_4_ = new uint8[size]();
    cost_aggr_5_ = new uint8[size]();
    cost_aggr_6_ = new uint8[size]();
    cost_aggr_7_ = new uint8[size]();
    cost_aggr_8_ = new uint8[size]();
    // 视差图
    disp_left_ = new float32[width * height]();

    is_initialized_ = census_left_ && census_right_ && cost_init_ && cost_aggr_ && disp_left_;

    return is_initialized_;
}

bool SemiGlobalMatching::Match(const uint8* img_left, const uint8* img_right, float32* disp_left)
{
    if (!is_initialized_) {
        return false;
    }
    if (img_left == nullptr || img_right == nullptr) {
        return false;
    }

    img_left_ = const_cast<uint8*> (img_left);
    img_right_ = const_cast<uint8*> (img_right);

    // census变换
    CensusTransform();

    // 代价计算
    ComputeCost();

    // 代价聚合
    CostAggregation();

    // 视差计算
    ComputeDisparity();


    // 输出视差图
    memcpy(disp_left, disp_left_, width_ * height_ * sizeof(float32));

    return true;
}
bool SemiGlobalMatching::Reset(const sint32& width, const sint32& height, const SGMOption& option)
{
    // 释放内存
    if (census_left_ != nullptr) {
        delete[] census_left_;
        census_left_ = nullptr;
    }
    if (census_right_ != nullptr) {
        delete[] census_right_;
        census_right_ = nullptr;
    }
    if (cost_init_ != nullptr) {
        delete[] cost_init_;
        cost_init_ = nullptr;
    }
    if (cost_aggr_ != nullptr) {
        delete[] cost_aggr_;
        cost_aggr_ = nullptr;
    }
    if (disp_left_ != nullptr) {
        delete[] disp_left_;
        disp_left_ = nullptr;
    }

    // 重置初始化标记
    is_initialized_ = false;

    // 初始化
    return Initialize(width, height, option);
}

void census_transform_5x5(const uint8* source, uint32* census, const sint32& width,
    const sint32& height)
{
    if (source == nullptr || census == nullptr || width <= 5u || height <= 5u) {
        return;
    }

    // 逐像素计算census值
    for (sint32 i = 2; i < height - 2; i++) {
        for (sint32 j = 2; j < width - 2; j++) {

            // 中心像素值
            const uint8 gray_center = source[i * width + j];

            // 遍历大小为5x5的窗口内邻域像素，逐一比较像素值与中心像素值的的大小，计算census值
            uint32 census_val = 0u;
            for (sint32 r = -2; r <= 2; r++) {
                for (sint32 c = -2; c <= 2; c++) {
                    census_val <<= 1;
                    const uint8 gray = source[(i + r) * width + j + c];
                    if (gray < gray_center) {
                        census_val += 1;
                    }
                }
            }

            // 中心像素的census值
            census[i * width + j] = census_val;
        }
    }
}

void SemiGlobalMatching::CensusTransform() const
{
    // 左右影像census变换
    census_transform_5x5(img_left_, census_left_, width_, height_);
    census_transform_5x5(img_right_, census_right_, width_, height_);
}
//计算hamming距离
uint16 Hamming32(const uint32& x, const uint32& y)
{
    uint32 dist = 0, val = x ^ y;

    // Count the number of set bits
    while (val) {
        ++dist;
        val &= val - 1;
    }

    return dist;
}
void SemiGlobalMatching::ComputeCost() const
{
    const sint32& min_disparity = option_.min_disparity;
    const sint32& max_disparity = option_.max_disparity;
    const sint32 disp_range = option_.max_disparity - option_.min_disparity;
    // 计算代价（基于Hamming距离）
    for (sint32 i = 0; i < height_; i++) {
        for (sint32 j = 0; j < width_; j++) {

            // 左影像census值
            const uint32 census_val_l = census_left_[i * width_ + j];

            // 逐视差计算代价值
            for (sint32 d = min_disparity; d < max_disparity; d++) {
                auto& cost = cost_init_[i * width_ * disp_range + j * disp_range + (d - min_disparity)];
                if (j - d < 0 || j - d >= width_) {
                    cost = UINT8_MAX / 2;
                    continue;
                }
                // 右影像对应像点的census值
                const uint32 census_val_r = census_right_[i * width_ + j - d];

                // 计算匹配代价
                cost = Hamming32(census_val_l, census_val_r);
            }
        }
    }
}
//视差计算
void SemiGlobalMatching::ComputeDisparity() const
{
    
    // 最小最大视差
    const sint32& min_disparity = option_.min_disparity;
    const sint32& max_disparity = option_.max_disparity;
    const sint32 disp_range = max_disparity - min_disparity;
    auto cost_ptr = cost_aggr_;

    // 逐像素计算最优视差
    for (sint32 i = 0; i < height_; i++) {
        for (sint32 j = 0; j < width_; j++) {

            uint16 min_cost = UINT16_MAX;
            uint16 max_cost = 0;
            sint32 best_disparity = 0;

            // 遍历视差范围内的所有代价值，输出最小代价值及对应的视差值
            for (sint32 d = min_disparity; d < max_disparity; d++) {
                const sint32 d_idx = d - min_disparity;
                const auto& cost = cost_ptr[i * width_ * disp_range + j * disp_range + d_idx];
                if (min_cost > cost) {
                    min_cost = cost;
                    best_disparity = d;
                }
                max_cost = std::max(max_cost, static_cast<uint16>(cost));
            }

            // 最小代价值对应的视差值即为像素的最优视差
            if (max_cost != min_cost) {
                disp_left_[i * width_ + j] = static_cast<float>(best_disparity);
            }
            else {
                // 如果所有视差下的代价值都一样，则该像素无效
                disp_left_[i * width_ + j] = 0;
            }
        }
    }
    
    
}

void CostAggregateLeftRight(const uint8* img_data, const sint32& width, const sint32& height, const sint32& min_disparity, const sint32& max_disparity,
    const sint32& p1, const sint32& p2_init, const uint8* cost_init, uint8* cost_aggr, bool is_forward)
{
    assert(width > 0 && height > 0 && max_disparity > min_disparity);

    // 视差范围
    const sint32 disp_range = max_disparity - min_disparity;

    // P1,P2
    const auto& P1 = p1;
    const auto& P2_Init = p2_init;

    // 正向(左->右) ：is_forward = true ; direction = 1
    // 反向(右->左) ：is_forward = false; direction = -1;
    const sint32 direction = is_forward ? 1 : -1;

    // 聚合
    for (sint32 i = 0u; i < height; i++) {
        // 路径头为每一行的首(尾,dir=-1)列像素
        auto cost_init_row = (is_forward) ? (cost_init + i * width * disp_range) : (cost_init + i * width * disp_range + (width - 1) * disp_range);
        auto cost_aggr_row = (is_forward) ? (cost_aggr + i * width * disp_range) : (cost_aggr + i * width * disp_range + (width - 1) * disp_range);
        auto img_row = (is_forward) ? (img_data + i * width) : (img_data + i * width + width - 1);

        // 路径上当前灰度值和上一个灰度值
        uint8 gray = *img_row;
        uint8 gray_last = *img_row;

        // 路径上上个像素的代价数组，多两个元素是为了避免边界溢出（首尾各多一个）
        std::vector<uint8> cost_last_path(disp_range + 2, UINT8_MAX);
        // 初始化：第一个像素的聚合代价值等于初始代价值
        memcpy(cost_aggr_row, cost_init_row, disp_range * sizeof(uint8));
        memcpy(&cost_last_path[1], cost_aggr_row, disp_range * sizeof(uint8));
        cost_init_row += direction * disp_range;
        cost_aggr_row += direction * disp_range;
        img_row += direction;

        // 路径上上个像素的最小代价值
        uint8 mincost_last_path = UINT8_MAX;
        for (auto cost : cost_last_path) {
            mincost_last_path = std::min(mincost_last_path, cost);
        }

        // 自方向上第2个像素开始按顺序聚合
        for (sint32 j = 0; j < width - 1; j++) {
            gray = *img_row;
            uint8 min_cost = UINT8_MAX;
            for (sint32 d = 0; d < disp_range; d++) {
                // Lr(p,d) = C(p,d) + min( Lr(p-r,d), Lr(p-r,d-1) + P1, Lr(p-r,d+1) + P1, min(Lr(p-r))+P2 ) - min(Lr(p-r))
                const uint8  cost = cost_init_row[d];
                const uint16 l1 = cost_last_path[d + 1];
                const uint16 l2 = cost_last_path[d] + P1;
                const uint16 l3 = cost_last_path[d + 2] + P1;
                const uint16 l4 = mincost_last_path + std::max(P1, P2_Init / (abs(gray - gray_last) + 1));

                const uint8 cost_s = cost + static_cast<uint8>(std::min(std::min(l1, l2), std::min(l3, l4)) - mincost_last_path);

                cost_aggr_row[d] = cost_s;
                min_cost = std::min(min_cost, cost_s);
            }

            // 重置上个像素的最小代价值和代价数组
            mincost_last_path = min_cost;
            memcpy(&cost_last_path[1], cost_aggr_row, disp_range * sizeof(uint8));

            // 下一个像素
            cost_init_row += direction * disp_range;
            cost_aggr_row += direction * disp_range;
            img_row += direction;

            // 像素值重新赋值
            gray_last = gray;
        }
    }
}

void CostAggregateUpDown(const uint8* img_data, const sint32& width, const sint32& height,
    const sint32& min_disparity, const sint32& max_disparity, const sint32& p1, const sint32& p2_init,
    const uint8* cost_init, uint8* cost_aggr, bool is_forward)
{
    assert(width > 0 && height > 0 && max_disparity > min_disparity);

    // 视差范围
    const sint32 disp_range = max_disparity - min_disparity;

    // P1,P2
    const auto& P1 = p1;
    const auto& P2_Init = p2_init;

    // 正向(上->下) ：is_forward = true ; direction = 1
    // 反向(下->上) ：is_forward = false; direction = -1;
    const sint32 direction = is_forward ? 1 : -1;

    // 聚合
    for (sint32 j = 0; j < width; j++) {
        // 路径头为每一列的首(尾,dir=-1)行像素
        auto cost_init_col = (is_forward) ? (cost_init + j * disp_range) : (cost_init + (height - 1) * width * disp_range + j * disp_range);
        auto cost_aggr_col = (is_forward) ? (cost_aggr + j * disp_range) : (cost_aggr + (height - 1) * width * disp_range + j * disp_range);
        auto img_col = (is_forward) ? (img_data + j) : (img_data + (height - 1) * width + j);

        // 路径上当前灰度值和上一个灰度值
        uint8 gray = *img_col;
        uint8 gray_last = *img_col;

        // 路径上上个像素的代价数组，多两个元素是为了避免边界溢出（首尾各多一个）
        std::vector<uint8> cost_last_path(disp_range + 2, UINT8_MAX);

        // 初始化：第一个像素的聚合代价值等于初始代价值
        memcpy(cost_aggr_col, cost_init_col, disp_range * sizeof(uint8));
        memcpy(&cost_last_path[1], cost_aggr_col, disp_range * sizeof(uint8));
        cost_init_col += direction * width * disp_range;
        cost_aggr_col += direction * width * disp_range;
        img_col += direction * width;

        // 路径上上个像素的最小代价值
        uint8 mincost_last_path = UINT8_MAX;
        for (auto cost : cost_last_path) {
            mincost_last_path = std::min(mincost_last_path, cost);
        }

        // 自方向上第2个像素开始按顺序聚合
        for (sint32 i = 0; i < height - 1; i++) {
            gray = *img_col;
            uint8 min_cost = UINT8_MAX;
            for (sint32 d = 0; d < disp_range; d++) {
                // Lr(p,d) = C(p,d) + min( Lr(p-r,d), Lr(p-r,d-1) + P1, Lr(p-r,d+1) + P1, min(Lr(p-r))+P2 ) - min(Lr(p-r))
                const uint8  cost = cost_init_col[d];
                const uint16 l1 = cost_last_path[d + 1];
                const uint16 l2 = cost_last_path[d] + P1;
                const uint16 l3 = cost_last_path[d + 2] + P1;
                const uint16 l4 = mincost_last_path + std::max(P1, P2_Init / (abs(gray - gray_last) + 1));

                const uint8 cost_s = cost + static_cast<uint8>(std::min(std::min(l1, l2), std::min(l3, l4)) - mincost_last_path);

                cost_aggr_col[d] = cost_s;
                min_cost = std::min(min_cost, cost_s);
            }

            // 重置上个像素的最小代价值和代价数组
            mincost_last_path = min_cost;
            memcpy(&cost_last_path[1], cost_aggr_col, disp_range * sizeof(uint8));

            // 下一个像素
            cost_init_col += direction * width * disp_range;
            cost_aggr_col += direction * width * disp_range;
            img_col += direction * width;

            // 像素值重新赋值
            gray_last = gray;
        }
    }
}

void SemiGlobalMatching::CostAggregation() const
{
    // 路径聚合
    // 1、左->右/右->左
    // 2、上->下/下->上
    // 3、左上->右下/右下->左上
    // 4、右上->左上/左下->右上
    //
    // ↘ ↓ ↙   5  3  7
    // →    ←	 1    2
    // ↗ ↑ ↖   8  4  6
    //
    const auto& min_disparity = option_.min_disparity;
    const auto& max_disparity = option_.max_disparity;
    assert(max_disparity > min_disparity);

    const sint32 size = width_ * height_ * (max_disparity - min_disparity);
    if (size <= 0) {
        return;
    }

    const auto& P1 = option_.p1;
    const auto& P2_Int = option_.p2_init;

    // 左右聚合
    CostAggregateLeftRight(img_left_, width_, height_, min_disparity, max_disparity, P1, P2_Int, cost_init_, cost_aggr_1_, true);
    CostAggregateLeftRight(img_left_, width_, height_, min_disparity, max_disparity, P1, P2_Int, cost_init_, cost_aggr_2_, false);
    // 上下聚合
    CostAggregateUpDown(img_left_, width_, height_, min_disparity, max_disparity, P1, P2_Int, cost_init_, cost_aggr_3_, true);
    CostAggregateUpDown(img_left_, width_, height_, min_disparity, max_disparity, P1, P2_Int, cost_init_, cost_aggr_4_, false);


    // 把4/8个方向加起来
    for (sint32 i = 0; i < size; i++) {
        cost_aggr_[i] = cost_aggr_1_[i] + cost_aggr_2_[i] + cost_aggr_3_[i] + cost_aggr_4_[i];
        if (option_.num_paths == 8) {
            cost_aggr_[i] += cost_aggr_5_[i] + cost_aggr_6_[i] + cost_aggr_7_[i] + cost_aggr_8_[i];
        }
    }
}

//均值滤波
void MeanFilter(const Mat& src, Mat& dst, int ksize)  //均值滤波   Scr 为要处理的图像，dst为目标图像，Ksize为卷积核的尺寸，卷积核的尺寸一般为 奇数
{
    CV_Assert(ksize % 2 == 1);                    // 不满足这个条件，则返回一个错误。           

    int* kernel = new int[ksize * ksize];           // 卷积核的大小
    for (int i = 0; i < ksize * ksize; i++)         // 均值滤波所以都为1
        kernel[i] = 1;
    Mat tmp;
    int len = ksize / 2;
    tmp.create(Size(src.cols + len, src.rows + len), src.type()); //添加边框
    dst.create(Size(src.cols, src.rows), src.type());

    int channel = src.channels();
    uchar* ps = src.data;
    uchar* pt = tmp.data;

    //添加边框是为了让图片周围的ksize/2 的像素都能进行均值滤波，若Ksize为3，,若是图片左上角的那个元素进行均值滤波，其实求的平均是 三个数(右、下、右下)的平均值。

    for (int row = 0; row < tmp.rows; row++)//添加边框的过程
    {
        for (int col = 0; col < tmp.cols; col++)
        {
            for (int c = 0; c < channel; c++)
            {
                if (row >= len && row < tmp.rows - len && col >= len && col < tmp.cols - len)
                    pt[(tmp.cols * row + col) * channel + c] = ps[(src.cols * (row - len) + col - len) * channel + c];
                else
                    pt[(tmp.cols * row + col) * channel + c] = 0;
            }
        }
    }


    uchar* pd = dst.data;
    pt = tmp.data;
    for (int row = len; row < tmp.rows - len; row++)//卷积的过程
    {
        for (int col = len; col < tmp.cols - len; col++)
        {
            for (int c = 0; c < channel; c++)
            {
                short t = 0;
                for (int x = -len; x <= len; x++)
                {
                    for (int y = -len; y <= len; y++)
                    {
                        t += kernel[(len + x) * ksize + y + len] * pt[((row + x) * tmp.cols + col + y) * channel + c];
                    }
                }
                pd[(dst.cols * (row - len) + col - len) * channel + c] = saturate_cast<ushort> (t / (ksize * ksize));//防止数据溢出ushort是16为数据
            }
        }
    }
    delete[] kernel;       // 释放 new 的卷积和空间
}

//中值滤波
Mat MedianFilter(const cv::Mat& src, int ksize)
{
    cv::Mat dst = src.clone();
    //获取图片的宽，高和像素信息，
    const int  num = ksize * ksize;
    std::vector<uchar> pixel(num);

    //相对于中心点，3*3领域中的点需要偏移的位置
    int delta[3 * 3][2] = {
        { -1, -1 }, { -1, 0 }, { -1, 1 }, { 0, -1 }, { 0, 0 }, { 0, 1 }, { 1, -1 }, { 1, 0 }, {1, 1}
    };
    //1. 中值滤波，没有考虑边缘
    for (int i = 1; i < src.rows - 1; ++i)
    {
        for (int j = 1; j < src.cols - 1; ++j)
        {
            //提取领域值 
            // 使用数组处理8邻域值,不适合更大窗口
            for (int k = 0; k < num; ++k)
            {
                pixel[k] = src.at<uchar>(i + delta[k][0], j + delta[k][1]);
            }
            //排序,使用自带的库
            std::sort(pixel.begin(), pixel.end());
            //获取该中心点的值
            dst.at<uchar>(i, j) = pixel[num / 2];
        }
    }
    return dst;
}
int main(int argv, char** argc)
{
    if (argv < 3) {
        // return 0;
    }

    //读取影像
    //std::string path_left = argc[1];
   // std::string path_right = argc[2];

    // SGM匹配参数设计
    SemiGlobalMatching::SGMOption sgm_option;
    // 聚合路径数
    // 唯一性约束
    //sgm_option.is_check_unique = true;
    //.uniqueness_ratio = 0.99;

    cv::Mat img_left = cv::imread("C://Users//86150//source//repos//SGMTEST//1.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img_right = cv::imread("C://Users//86150//source//repos//SGMTEST//2.png", cv::IMREAD_GRAYSCALE);

    if (img_left.data == nullptr || img_right.data == nullptr) {
        std::cout << "读取影像失败！" << std::endl;
        return -1;
    }
    if (img_left.rows != img_right.rows || img_left.cols != img_right.cols) {
        std::cout << "左右影像尺寸不一致！" << std::endl;
        return -1;
    }

    //SGM匹配
    const uint32 width = static_cast<uint32>(img_left.cols);
    const uint32 height = static_cast<uint32>(img_right.rows);

    //SemiGlobalMatching::SGMOption sgm_option;
    sgm_option.num_paths = 8;
    sgm_option.min_disparity = 0;
    sgm_option.max_disparity = 64;
    sgm_option.p1 = 10;
    sgm_option.p2_init = 150;

    SemiGlobalMatching sgm;

    // 初始化
    if (!sgm.Initialize(width, height, sgm_option)) {
        std::cout << "SGM初始化失败！" << std::endl;
        return -2;
    }




    // 匹配
    auto disparity = new float32[width * height]();
    if (!sgm.Match(img_left.data, img_right.data, disparity)) {
        std::cout << "SGM匹配失败！" << std::endl;
        return -2;
    }

    // 显示视差图
    cv::Mat disp_mat = cv::Mat(height, width, CV_8UC1);
    for (uint32 i = 0; i < height; i++) {
        for (uint32 j = 0; j < width; j++) {
            const float32 disp = disparity[i * width + j];
            if (disp == 0) {
                disp_mat.data[i * width + j] = 0;
            }
            else {
                disp_mat.data[i * width + j] = 2 * static_cast<uchar>(disp);
            }
        }
    }


    cv::imwrite("5.png", disp_mat);
    cv::imshow("视差图", disp_mat);

    //手写均值滤波
    /*Mat src = imread("5.png");
    Mat dst = Mat::zeros(src.size(), src.type());
    int k = 7;
    cv::imwrite("meanfilter.png", dst);
    MeanFilter(src, dst, k);
    imshow("meanfilter", dst);*/
    //试了高斯滤波，但是没什么作用
    
    //手写中值滤波
    //cv::Mat srcImage = disp_mat;
    //cv::Mat dstImage;
    //MedianFilter(srcImage,3);//窗口大小ksize必须为奇数
    //cv::imshow("medianfilter.jpg", srcImage);

    //直接调用opcencv函数中值滤波
    cv::Mat srcImage = disp_mat;
    cv::Mat dst;
    medianBlur(srcImage, dst, 5);
    ////cv::imwrite("medianBlured.png", dst);
    cv::imshow("medianblured.jpg", dst);
    
    
    cv::waitKey(0);
    system("pause");

    delete[] disparity;
    disparity = nullptr;

    return 0;
}




