//
// Created by zbox on 18-9-7.
//

#ifndef MYSLAM_VISUAL_ODOMETRY_H
#define MYSLAM_VISUAL_ODOMETRY_H

#include "myslam/common_include.h"
#include "myslam/map.h"
#include <opencv2/features2d/features2d.hpp>

namespace myslam
{
    class VisualOdometry
    {
    public:
        typedef shared_ptr<VisualOdometry> Ptr;
        enum VOState
        {
            INITIALIZING = -1,
                    OK,
            LOST
        };

        VOState     state_;     // current VO status
        Map::Ptr    map_;       // map with all frames and map points

        Frame::Ptr  ref_;       // reference key-frame
        Frame::Ptr  curr_;      // current frame

        cv::Ptr<cv::ORB>    orb_;       // orb detector and computer
        vector<cv::KeyPoint>    keypoints_curr_;    // keypoints in current frame
        Mat                     descriptors_curr_;  // descriptor in current frame

        // descriptor in reference frame, be matched with descriptors_curr_ to find matched points between ref and curr frame
        //Mat                     descriptors_ref_; // in this project, current frame is matched with nearby map points

        cv::FlannBasedMatcher   matcher_flann_;     // flann matcher
        vector<MapPoint::Ptr>   match_3dpts_;       // matched 3d points
        vector<int>             match_2dkp_index_;   // matched 2d pixels (index of kp_curr)

        SE3 T_c_w_estimated_;       // the estimated pose of current frame
        int num_inliers_;           // number of inlier features in icp
        int num_lost_;              // number of lost times

        // parameters
        int num_of_features_;       // number of features
        double scale_factor_;       // scale in image pyramid
        int level_pyramid_;         // number of pyramid levels
        float match_ratio_;         // ratio for selecting  good matches
        int max_num_lost_;          // max number of continuous lost times
        int min_inliers_;           // minimum inliers
        double key_frame_min_rot_;  // minimal rotation of two key-frames
        double key_frame_min_trans_;    //minimal translation of two key-frames
        double map_point_erase_ratio_;  // remove map point ratio, update map

    public: // functions
        VisualOdometry();
        ~VisualOdometry();

        bool addFrame(Frame::Ptr frame);    // add a new frame

    protected:
        // inner operation
        void extractKeyPoints();
        void computeDescriptors();
        void featureMatching();
        void poseEstimationPnP();
        void optimizeMap();

        void addKeyFrame();
        void addMapPoints();
        bool checkEstimatedPose();
        bool checkKeyFrame();

        double getViewAngle(Frame::Ptr frame, MapPoint::Ptr point);
    };
}


#endif //MYSLAM_VISUAL_ODOMETRY_H
