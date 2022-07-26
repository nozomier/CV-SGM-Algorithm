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

    // Ӱ��ߴ�
    width_ = width;
    height_ = height;
    // SGM����
    option_ = option;

    if (width == 0 || height == 0) {
        return false;
    }

    // �����ڴ�ռ�

    // censusֵ������Ӱ��
    census_left_ = new uint32[width * height]();
    census_right_ = new uint32[width * height]();

    // ƥ����ۣ���ʼ/�ۺϣ�
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
    // �Ӳ�ͼ
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

    // census�任
    CensusTransform();

    // ���ۼ���
    ComputeCost();

    // ���۾ۺ�
    CostAggregation();

    // �Ӳ����
    ComputeDisparity();


    // ����Ӳ�ͼ
    memcpy(disp_left, disp_left_, width_ * height_ * sizeof(float32));

    return true;
}
bool SemiGlobalMatching::Reset(const sint32& width, const sint32& height, const SGMOption& option)
{
    // �ͷ��ڴ�
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

    // ���ó�ʼ�����
    is_initialized_ = false;

    // ��ʼ��
    return Initialize(width, height, option);
}

void census_transform_5x5(const uint8* source, uint32* census, const sint32& width,
    const sint32& height)
{
    if (source == nullptr || census == nullptr || width <= 5u || height <= 5u) {
        return;
    }

    // �����ؼ���censusֵ
    for (sint32 i = 2; i < height - 2; i++) {
        for (sint32 j = 2; j < width - 2; j++) {

            // ��������ֵ
            const uint8 gray_center = source[i * width + j];

            // ������СΪ5x5�Ĵ������������أ���һ�Ƚ�����ֵ����������ֵ�ĵĴ�С������censusֵ
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

            // �������ص�censusֵ
            census[i * width + j] = census_val;
        }
    }
}

void SemiGlobalMatching::CensusTransform() const
{
    // ����Ӱ��census�任
    census_transform_5x5(img_left_, census_left_, width_, height_);
    census_transform_5x5(img_right_, census_right_, width_, height_);
}
//����hamming����
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
    // ������ۣ�����Hamming���룩
    for (sint32 i = 0; i < height_; i++) {
        for (sint32 j = 0; j < width_; j++) {

            // ��Ӱ��censusֵ
            const uint32 census_val_l = census_left_[i * width_ + j];

            // ���Ӳ�������ֵ
            for (sint32 d = min_disparity; d < max_disparity; d++) {
                auto& cost = cost_init_[i * width_ * disp_range + j * disp_range + (d - min_disparity)];
                if (j - d < 0 || j - d >= width_) {
                    cost = UINT8_MAX / 2;
                    continue;
                }
                // ��Ӱ���Ӧ����censusֵ
                const uint32 census_val_r = census_right_[i * width_ + j - d];

                // ����ƥ�����
                cost = Hamming32(census_val_l, census_val_r);
            }
        }
    }
}
//�Ӳ����
void SemiGlobalMatching::ComputeDisparity() const
{
    
    // ��С����Ӳ�
    const sint32& min_disparity = option_.min_disparity;
    const sint32& max_disparity = option_.max_disparity;
    const sint32 disp_range = max_disparity - min_disparity;
    auto cost_ptr = cost_aggr_;

    // �����ؼ��������Ӳ�
    for (sint32 i = 0; i < height_; i++) {
        for (sint32 j = 0; j < width_; j++) {

            uint16 min_cost = UINT16_MAX;
            uint16 max_cost = 0;
            sint32 best_disparity = 0;

            // �����ӲΧ�ڵ����д���ֵ�������С����ֵ����Ӧ���Ӳ�ֵ
            for (sint32 d = min_disparity; d < max_disparity; d++) {
                const sint32 d_idx = d - min_disparity;
                const auto& cost = cost_ptr[i * width_ * disp_range + j * disp_range + d_idx];
                if (min_cost > cost) {
                    min_cost = cost;
                    best_disparity = d;
                }
                max_cost = std::max(max_cost, static_cast<uint16>(cost));
            }

            // ��С����ֵ��Ӧ���Ӳ�ֵ��Ϊ���ص������Ӳ�
            if (max_cost != min_cost) {
                disp_left_[i * width_ + j] = static_cast<float>(best_disparity);
            }
            else {
                // ��������Ӳ��µĴ���ֵ��һ�������������Ч
                disp_left_[i * width_ + j] = 0;
            }
        }
    }
    
    
}

void CostAggregateLeftRight(const uint8* img_data, const sint32& width, const sint32& height, const sint32& min_disparity, const sint32& max_disparity,
    const sint32& p1, const sint32& p2_init, const uint8* cost_init, uint8* cost_aggr, bool is_forward)
{
    assert(width > 0 && height > 0 && max_disparity > min_disparity);

    // �ӲΧ
    const sint32 disp_range = max_disparity - min_disparity;

    // P1,P2
    const auto& P1 = p1;
    const auto& P2_Init = p2_init;

    // ����(��->��) ��is_forward = true ; direction = 1
    // ����(��->��) ��is_forward = false; direction = -1;
    const sint32 direction = is_forward ? 1 : -1;

    // �ۺ�
    for (sint32 i = 0u; i < height; i++) {
        // ·��ͷΪÿһ�е���(β,dir=-1)������
        auto cost_init_row = (is_forward) ? (cost_init + i * width * disp_range) : (cost_init + i * width * disp_range + (width - 1) * disp_range);
        auto cost_aggr_row = (is_forward) ? (cost_aggr + i * width * disp_range) : (cost_aggr + i * width * disp_range + (width - 1) * disp_range);
        auto img_row = (is_forward) ? (img_data + i * width) : (img_data + i * width + width - 1);

        // ·���ϵ�ǰ�Ҷ�ֵ����һ���Ҷ�ֵ
        uint8 gray = *img_row;
        uint8 gray_last = *img_row;

        // ·�����ϸ����صĴ������飬������Ԫ����Ϊ�˱���߽��������β����һ����
        std::vector<uint8> cost_last_path(disp_range + 2, UINT8_MAX);
        // ��ʼ������һ�����صľۺϴ���ֵ���ڳ�ʼ����ֵ
        memcpy(cost_aggr_row, cost_init_row, disp_range * sizeof(uint8));
        memcpy(&cost_last_path[1], cost_aggr_row, disp_range * sizeof(uint8));
        cost_init_row += direction * disp_range;
        cost_aggr_row += direction * disp_range;
        img_row += direction;

        // ·�����ϸ����ص���С����ֵ
        uint8 mincost_last_path = UINT8_MAX;
        for (auto cost : cost_last_path) {
            mincost_last_path = std::min(mincost_last_path, cost);
        }

        // �Է����ϵ�2�����ؿ�ʼ��˳��ۺ�
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

            // �����ϸ����ص���С����ֵ�ʹ�������
            mincost_last_path = min_cost;
            memcpy(&cost_last_path[1], cost_aggr_row, disp_range * sizeof(uint8));

            // ��һ������
            cost_init_row += direction * disp_range;
            cost_aggr_row += direction * disp_range;
            img_row += direction;

            // ����ֵ���¸�ֵ
            gray_last = gray;
        }
    }
}

void CostAggregateUpDown(const uint8* img_data, const sint32& width, const sint32& height,
    const sint32& min_disparity, const sint32& max_disparity, const sint32& p1, const sint32& p2_init,
    const uint8* cost_init, uint8* cost_aggr, bool is_forward)
{
    assert(width > 0 && height > 0 && max_disparity > min_disparity);

    // �ӲΧ
    const sint32 disp_range = max_disparity - min_disparity;

    // P1,P2
    const auto& P1 = p1;
    const auto& P2_Init = p2_init;

    // ����(��->��) ��is_forward = true ; direction = 1
    // ����(��->��) ��is_forward = false; direction = -1;
    const sint32 direction = is_forward ? 1 : -1;

    // �ۺ�
    for (sint32 j = 0; j < width; j++) {
        // ·��ͷΪÿһ�е���(β,dir=-1)������
        auto cost_init_col = (is_forward) ? (cost_init + j * disp_range) : (cost_init + (height - 1) * width * disp_range + j * disp_range);
        auto cost_aggr_col = (is_forward) ? (cost_aggr + j * disp_range) : (cost_aggr + (height - 1) * width * disp_range + j * disp_range);
        auto img_col = (is_forward) ? (img_data + j) : (img_data + (height - 1) * width + j);

        // ·���ϵ�ǰ�Ҷ�ֵ����һ���Ҷ�ֵ
        uint8 gray = *img_col;
        uint8 gray_last = *img_col;

        // ·�����ϸ����صĴ������飬������Ԫ����Ϊ�˱���߽��������β����һ����
        std::vector<uint8> cost_last_path(disp_range + 2, UINT8_MAX);

        // ��ʼ������һ�����صľۺϴ���ֵ���ڳ�ʼ����ֵ
        memcpy(cost_aggr_col, cost_init_col, disp_range * sizeof(uint8));
        memcpy(&cost_last_path[1], cost_aggr_col, disp_range * sizeof(uint8));
        cost_init_col += direction * width * disp_range;
        cost_aggr_col += direction * width * disp_range;
        img_col += direction * width;

        // ·�����ϸ����ص���С����ֵ
        uint8 mincost_last_path = UINT8_MAX;
        for (auto cost : cost_last_path) {
            mincost_last_path = std::min(mincost_last_path, cost);
        }

        // �Է����ϵ�2�����ؿ�ʼ��˳��ۺ�
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

            // �����ϸ����ص���С����ֵ�ʹ�������
            mincost_last_path = min_cost;
            memcpy(&cost_last_path[1], cost_aggr_col, disp_range * sizeof(uint8));

            // ��һ������
            cost_init_col += direction * width * disp_range;
            cost_aggr_col += direction * width * disp_range;
            img_col += direction * width;

            // ����ֵ���¸�ֵ
            gray_last = gray;
        }
    }
}

void SemiGlobalMatching::CostAggregation() const
{
    // ·���ۺ�
    // 1����->��/��->��
    // 2����->��/��->��
    // 3������->����/����->����
    // 4������->����/����->����
    //
    // �K �� �L   5  3  7
    // ��    ��	 1    2
    // �J �� �I   8  4  6
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

    // ���Ҿۺ�
    CostAggregateLeftRight(img_left_, width_, height_, min_disparity, max_disparity, P1, P2_Int, cost_init_, cost_aggr_1_, true);
    CostAggregateLeftRight(img_left_, width_, height_, min_disparity, max_disparity, P1, P2_Int, cost_init_, cost_aggr_2_, false);
    // ���¾ۺ�
    CostAggregateUpDown(img_left_, width_, height_, min_disparity, max_disparity, P1, P2_Int, cost_init_, cost_aggr_3_, true);
    CostAggregateUpDown(img_left_, width_, height_, min_disparity, max_disparity, P1, P2_Int, cost_init_, cost_aggr_4_, false);


    // ��4/8�����������
    for (sint32 i = 0; i < size; i++) {
        cost_aggr_[i] = cost_aggr_1_[i] + cost_aggr_2_[i] + cost_aggr_3_[i] + cost_aggr_4_[i];
        if (option_.num_paths == 8) {
            cost_aggr_[i] += cost_aggr_5_[i] + cost_aggr_6_[i] + cost_aggr_7_[i] + cost_aggr_8_[i];
        }
    }
}

//��ֵ�˲�
void MeanFilter(const Mat& src, Mat& dst, int ksize)  //��ֵ�˲�   Scr ΪҪ�����ͼ��dstΪĿ��ͼ��KsizeΪ����˵ĳߴ磬����˵ĳߴ�һ��Ϊ ����
{
    CV_Assert(ksize % 2 == 1);                    // ����������������򷵻�һ������           

    int* kernel = new int[ksize * ksize];           // ����˵Ĵ�С
    for (int i = 0; i < ksize * ksize; i++)         // ��ֵ�˲����Զ�Ϊ1
        kernel[i] = 1;
    Mat tmp;
    int len = ksize / 2;
    tmp.create(Size(src.cols + len, src.rows + len), src.type()); //��ӱ߿�
    dst.create(Size(src.cols, src.rows), src.type());

    int channel = src.channels();
    uchar* ps = src.data;
    uchar* pt = tmp.data;

    //��ӱ߿���Ϊ����ͼƬ��Χ��ksize/2 �����ض��ܽ��о�ֵ�˲�����KsizeΪ3��,����ͼƬ���Ͻǵ��Ǹ�Ԫ�ؽ��о�ֵ�˲�����ʵ���ƽ���� ������(�ҡ��¡�����)��ƽ��ֵ��

    for (int row = 0; row < tmp.rows; row++)//��ӱ߿�Ĺ���
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
    for (int row = len; row < tmp.rows - len; row++)//����Ĺ���
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
                pd[(dst.cols * (row - len) + col - len) * channel + c] = saturate_cast<ushort> (t / (ksize * ksize));//��ֹ�������ushort��16Ϊ����
            }
        }
    }
    delete[] kernel;       // �ͷ� new �ľ���Ϳռ�
}

//��ֵ�˲�
Mat MedianFilter(const cv::Mat& src, int ksize)
{
    cv::Mat dst = src.clone();
    //��ȡͼƬ�Ŀ��ߺ�������Ϣ��
    const int  num = ksize * ksize;
    std::vector<uchar> pixel(num);

    //��������ĵ㣬3*3�����еĵ���Ҫƫ�Ƶ�λ��
    int delta[3 * 3][2] = {
        { -1, -1 }, { -1, 0 }, { -1, 1 }, { 0, -1 }, { 0, 0 }, { 0, 1 }, { 1, -1 }, { 1, 0 }, {1, 1}
    };
    //1. ��ֵ�˲���û�п��Ǳ�Ե
    for (int i = 1; i < src.rows - 1; ++i)
    {
        for (int j = 1; j < src.cols - 1; ++j)
        {
            //��ȡ����ֵ 
            // ʹ�����鴦��8����ֵ,���ʺϸ��󴰿�
            for (int k = 0; k < num; ++k)
            {
                pixel[k] = src.at<uchar>(i + delta[k][0], j + delta[k][1]);
            }
            //����,ʹ���Դ��Ŀ�
            std::sort(pixel.begin(), pixel.end());
            //��ȡ�����ĵ��ֵ
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

    //��ȡӰ��
    //std::string path_left = argc[1];
   // std::string path_right = argc[2];

    // SGMƥ��������
    SemiGlobalMatching::SGMOption sgm_option;
    // �ۺ�·����
    // Ψһ��Լ��
    //sgm_option.is_check_unique = true;
    //.uniqueness_ratio = 0.99;

    cv::Mat img_left = cv::imread("C://Users//86150//source//repos//SGMTEST//1.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img_right = cv::imread("C://Users//86150//source//repos//SGMTEST//2.png", cv::IMREAD_GRAYSCALE);

    if (img_left.data == nullptr || img_right.data == nullptr) {
        std::cout << "��ȡӰ��ʧ�ܣ�" << std::endl;
        return -1;
    }
    if (img_left.rows != img_right.rows || img_left.cols != img_right.cols) {
        std::cout << "����Ӱ��ߴ粻һ�£�" << std::endl;
        return -1;
    }

    //SGMƥ��
    const uint32 width = static_cast<uint32>(img_left.cols);
    const uint32 height = static_cast<uint32>(img_right.rows);

    //SemiGlobalMatching::SGMOption sgm_option;
    sgm_option.num_paths = 8;
    sgm_option.min_disparity = 0;
    sgm_option.max_disparity = 64;
    sgm_option.p1 = 10;
    sgm_option.p2_init = 150;

    SemiGlobalMatching sgm;

    // ��ʼ��
    if (!sgm.Initialize(width, height, sgm_option)) {
        std::cout << "SGM��ʼ��ʧ�ܣ�" << std::endl;
        return -2;
    }




    // ƥ��
    auto disparity = new float32[width * height]();
    if (!sgm.Match(img_left.data, img_right.data, disparity)) {
        std::cout << "SGMƥ��ʧ�ܣ�" << std::endl;
        return -2;
    }

    // ��ʾ�Ӳ�ͼ
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
    cv::imshow("�Ӳ�ͼ", disp_mat);

    //��д��ֵ�˲�
    /*Mat src = imread("5.png");
    Mat dst = Mat::zeros(src.size(), src.type());
    int k = 7;
    cv::imwrite("meanfilter.png", dst);
    MeanFilter(src, dst, k);
    imshow("meanfilter", dst);*/
    //���˸�˹�˲�������ûʲô����
    
    //��д��ֵ�˲�
    //cv::Mat srcImage = disp_mat;
    //cv::Mat dstImage;
    //MedianFilter(srcImage,3);//���ڴ�Сksize����Ϊ����
    //cv::imshow("medianfilter.jpg", srcImage);

    //ֱ�ӵ���opcencv������ֵ�˲�
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




