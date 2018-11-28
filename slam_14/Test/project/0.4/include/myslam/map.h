//
// Created by zbox on 18-9-6.
//

#ifndef MYSLAM_MAP_H
#define MYSLAM_MAP_H

#include "myslam/common_include.h"
#include "myslam/mappoint.h"
#include "myslam/frame.h"

namespace myslam
{
    class Map
    {
    public:
        typedef shared_ptr<Map> Ptr;
        unordered_map<unsigned long, MapPoint::Ptr>     map_points_;    // all landmarks
        unordered_map<unsigned long, Frame::Ptr>        keyframes_;     // all key-frames

        Map()   {}  // default constructor

        void insertKeyFrame(Frame::Ptr frame);
        void insertMapPoint(MapPoint::Ptr map_point);
    };
}

#endif //MYSLAM_MAP_H
