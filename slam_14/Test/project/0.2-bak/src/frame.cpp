//
// Created by zbox on 18-8-24.
//

#include "myslam/frame.h"

namespace myslam
{
    Frame::Frame()  // constructor
            : id_(-1), time_stamp_(-1), camera_(nullptr)
    {

    }

    // constructor with initialization list
    Frame::Frame(long id, double time_stamp, SE3 T_c_w, Camera::Ptr camera, Mat color, Mat depth )
            : id_(id), time_stamp_(time_stamp), T_c_w_(T_c_w), camera_(camera), color_(color), depth_(depth)
    {

    }

    Frame::~Frame()
    {

    }

    Frame::Ptr Frame::createFrame()
    {
        static long factory_id = 0; // factory_id 设置为静态的，记录创建的frame数目
        return Frame::Ptr(new Frame(factory_id));   // which constructor function is utilized
    }

    double Frame::findDepth(const cv::KeyPoint &kp)
    {
        int x = cvRound(kp.pt.x);
        int y = cvRound(kp.pt.y);
        ushort d = depth_.ptr<ushort >(y)[x];
        if (0 != d)
        {
            return double(d) / camera_->depth_scale_;
        }
        else
        {
            // check nearby points
            int dx[4] = {-1, 0, 1, 0};
            int dy[4] = {0, -1, 0, 1};
            for (int i = 0; i < 4; ++i)
            {
                d = depth_.ptr<ushort >(y+dy[i])[x+dx[i]];
                if (0 != d)
                {
                    return double(d) / camera_->depth_scale_;
                }
            }
        }

        return -1;
    }


    Vector3d Frame::getCamCenter() const
    {
        return T_c_w_.inverse().translation();   // T_c_w的取逆后平移矩阵，代表了相机光心相对位置的平移，而旋转对一个光心是不起作用的
    }

    bool Frame::isInFrame(const Vector3d &pt_world)
    {
        Vector3d p_cam = camera_->world2camera(pt_world, T_c_w_);
        if (p_cam(2,0) < 0) // 投影的距离总不能为负数吧
            return false;
        Vector2d pixel = camera_->camera2pixel(p_cam);
        return pixel(0,0) > 0 && pixel(1,0) > 0
                && pixel(0,0) < color_.cols
                && pixel(1,0) < color_.rows;  // 在像素平面的成像点不超过图像大小的边界
    }
}