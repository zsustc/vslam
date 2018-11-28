//
// Created by zbox on 18-9-6.
//

#include "myslam/map.h"
namespace myslam
{
   // function definitions for class map
    void Map::insertKeyFrame(myslam::Frame::Ptr frame)
   {
       cout << "Key frame size = " << keyframes_.size() << endl;
       if (keyframes_.find(frame->id_) == keyframes_.end())
       {
           keyframes_.insert(make_pair(frame->id_, frame));
       }
       else
           {
               keyframes_[frame->id_] = frame;
           }

   }

   void Map::insertMapPoint(myslam::MapPoint::Ptr map_point)
   {
       // 判断map_points中是否已经存在该id的点：第一种情况，代表不存在，则在末尾插入
       // 第二种情况就是已经存在该id的点，则更新原先存在的点
       if (map_points_.find(map_point->id_) == map_points_.end())
       {
           map_points_.insert(make_pair(map_point->id_, map_point));
       }
       else
           {
               map_points_[map_point->id_] = map_point;
           }
   }
}

