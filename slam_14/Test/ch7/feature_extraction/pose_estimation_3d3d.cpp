//
// Created by nuc on 18-7-7.
//
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <chrono>

using namespace std;
using namespace cv;

// 寻找两幅图片的ORB特征匹配对
void find_feature_matches( const Mat& img_1, const Mat& img_2,
                           std::vector<KeyPoint>& keypoints_1,
                           std::vector<KeyPoint>& keypoints_2,
                           std::vector<DMatch>& matches );
// 将像素坐标转换到相机归一化坐标
Point2f pixel2cam( const Point2d& p, const Mat& K );

// 计算两组3D点的位姿变换
void pose_estimation_3d3d(
        const vector<Point3f>& pts1,
        const vector<Point3f>& pts2,
        Mat& R, Mat& t
);

int main(int argc, char** argv)
{
    if (5 != argc )
    {
        cout<<"usage: pose_estimation_3d3d img1 img2 depth1 depth2"<<endl;
        return 1;
    }

    Mat img_1 = imread( argv[1], CV_LOAD_IMAGE_COLOR );
    Mat img_2 = imread( argv[2], CV_LOAD_IMAGE_COLOR );

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches( img_1, img_2, keypoints_1, keypoints_2, matches );
    cout<<"一共找到了"<<matches.size()<<"组匹配点"<<endl;

    // 建立3D点
    Mat depth1 = imread( argv[3], CV_LOAD_IMAGE_UNCHANGED ); // 深度图为16位无符号数，单通道
    Mat depth2 = imread( argv[4], CV_LOAD_IMAGE_UNCHANGED );




}





