//
// Created by zbox on 18-9-6.
//

#ifndef MYSLAM_FRAME_H
#define MYSLAM_FRAME_H

#include "myslam/common_include.h"
#include "myslam/camera.h"
#include "myslam/mappoint.h"

namespace myslam
{
    class MapPoint;
    class Frame
    {
    public:
        typedef std::shared_ptr<Frame>   Ptr;
        unsigned long                id_;   // id of this frame
        double                      time_stamp_;
        SE3                         T_c_w_; // transform from world to camera
        Camera::Ptr                 camera_;    // Pinhole RGBD Camera model
        Mat                         color_,depth_;  // color and depth image
        std::vector<cv::KeyPoint>   keypoints_;     // key points in image
        std::vector<MapPoint>       map_points_;    // associated map points
        bool                        is_key_frame_;  // whether a key frame

    public:
        Frame();
        Frame(long id, double time_stamp = 0, SE3 T_c_w = SE3(), Camera::Ptr camera = nullptr, Mat color = Mat(), Mat depth = Mat());
        ~Frame();

        static Frame::Ptr createFrame();    // this is a static function, which can be called without creating a class object

        // find the depth in depth map
        double findDepth(const cv::KeyPoint& kp);

        // Get Camera Center
        Vector3d getCamCenter() const;

        void setPose(const SE3& T_c_w);

        // check if a point is in this frame,
        // point in world system is transformed into pixel system and check whether it is located at pixel plane
        bool isInframe(const Vector3d& pt_world);
    };

}

#endif //MYSLAM_FRAME_H
