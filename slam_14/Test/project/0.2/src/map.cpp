//
// Created by zbox on 18-8-24.
//

// hash 存储，支持随时存储、随时插入和删除
#include <iostream>
#include "myslam/map.h"

namespace myslam
{
    void Map::insertKeyFrame(Frame::Ptr frame)
    {
        cout << "key frame size = " << keyframes_.size() << endl;
        if (keyframes_.find(frame->id_) == keyframes_.end())
        {
            keyframes_.insert(make_pair(frame->id_, frame));    // 新的frame
            cout << "frame->id: " << frame->id_ << endl;
            cout << "frame->T_c_w" << frame->T_c_w_.rotation_matrix()<<endl;
            cout << "frame->T_c_w" << frame->T_c_w_.translation()<<endl;
        }
        else
        {
            keyframes_[frame->id_] = frame; // 更新已有的frame
        }
    }

    void Map::insertMapPoint(MapPoint::Ptr map_point)
    {
        if (map_points_.find(map_point->id_) == map_points_.end())
        {
            map_points_.insert(make_pair(map_point->id_, map_point));
        }
        else
        {
            map_points_[map_point->id_] =map_point;
        }
    }
}