//
// Created by zbox on 18-8-24.
// 表示路标点
//

#ifndef MYSLAM_MAPPOINT_H
#define MYSLAM_MAPPOINT_H

#include "myslam/common_include.h"

namespace myslam
{
    class MapPoint
    {
    public:
        typedef std::shared_ptr<MapPoint> Ptr;
        unsigned long       id_;    // ID of this map point
        Vector3d        pos_;       // position in world
        Vector3d        norm_;      // Norm of viewing direction ?这是什么
        Mat             descriptor_;    // Descriptor for matching 在图像中提取到对应位置的特征吗
        int             observed_times_;     // being observed by feature matching algorithm
        int             correct_times_;      // being an inlier in pose estimation

        MapPoint();
        MapPoint(long id, Vector3d position, Vector3d norm);

        // factory function
        static MapPoint::Ptr createMapPoint();
    };
}

#endif //MYSLAM_MAPPOINT_H
