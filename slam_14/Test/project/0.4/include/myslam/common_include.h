//
// Created by zbox on 18-9-6.
//

#ifndef MYSLAM_COMMON_INCLUDE_H
#define MYSLAM_COMMON_INCLUDE_H

// define the commonly included file to avoid a long include list
// for eigen
#include <Eigen/Core>
#include <Eigen/Geometry>
using Eigen::Vector3d;
using Eigen::Vector2d;

// for sophus
#include <sophus/se3.h>
#include <sophus/so3.h>
using Sophus::SO3;
using Sophus::SE3;

// for cv
#include <opencv2/core/core.hpp>
using cv::Mat;

// std
#include <vector>
#include <list>
#include <memory>
#include <string>
#include <iostream>
#include <set>
#include <unordered_map>
#include <map>

using namespace std;

#endif //MYSLAM_COMMON_INCLUDE_H
