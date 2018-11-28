//
// Created by zbox on 18-9-6.
//

#include "myslam/frame.h"

namespace myslam
{

   Frame::Frame()
           :id_(-1), time_stamp_(-1), camera_(nullptr)
   {

   }


    Frame::Frame(long id, double time_stamp, SE3 T_c_w, Camera::Ptr camera, Mat color, Mat depth)
            : id_(id), time_stamp_(time_stamp), T_c_w_(T_c_w), camera_(camera), color_(color), depth_(depth)
    {

    }

    Frame::~Frame()
    {

    }

    Frame::Ptr Frame::createFrame()
    {
        static long factory_id = 0; // 静态变量，具有记忆性，在类对象没有销毁前，一直存在
        return Frame::Ptr(new Frame(factory_id++));
    }


    // color图已经detect出keypoint了，去寻找keypoint对应的depth
    double Frame::findDepth(const cv::KeyPoint &kp)
    {
        int x = cvRound(kp.pt.x);   // kp.pt.x 可能会不是整形数吗
        int y = cvRound(kp.pt.y);

        // ushort d = depth_.at<ushort>(kp.pt.x, kp.pt.y);
        ushort d = depth_.ptr<ushort>(y)[x];
        if (d != 0)
        {
            return double(d)/camera_->depth_scale_;
        } else
        {
            int dx[4] = {-1,0,1,0};
            int dy[4] = {0,-1,0,1};

            for (int i = 0; i < 4; ++i)
            {
                d = depth_.ptr<ushort>(y + dy[i])[x + dx[i]];
                if (0 != d)
                {
                    return double(d) / camera_->depth_scale_;
                }
            }
        }

        return -1.0;
    }

    void Frame::setPose(const SE3 &T_c_w)
    {
        T_c_w_ = T_c_w;
    }

    bool Frame::isInframe(const Vector3d &pt_world)
    {
        Vector3d p_cam = camera_->world2camera(pt_world, T_c_w_);
        if(p_cam(2,0) < 0)      //求得相机坐标系坐标，深度信息竟然为负的，什么情况会出现呢
            return false;

        Vector2d pixel = camera_->camera2pixel(p_cam);
        // u = pixel(0,0), v = pixel(1,0); u,v 分别是像素点的x,y轴的坐标， x,y的值都大于0且小于color的行列
        return pixel(0,0) > 0 && pixel(1,0) > 0
                && pixel(0,0) < color_.cols
                && pixel(1,0) < color_.rows;
    }

    // 相机坐标系下的(0,0,0)在世界坐标系下的坐标，也就是相机光心的世界坐标
    Vector3d Frame::getCamCenter() const
    {
        return T_c_w_.inverse().translation();
    }

}