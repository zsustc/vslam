//
// Created by zbox on 18-9-7.
//


/*
 * 全局 3D-2D 点匹配算法 3D点是转化到第一帧帧图像 相机坐标系(世界坐标系)下的点
 * 匹配算法 也是当前描述子和  地图描述子匹配 对应的 3D点
 * 地图点 第一帧 的3D点全部加入地图 此后 新的一帧 图像在地图中 没有匹配到的 像素点 转换成世界坐标系下的三维点后 加入地图
 * 同时对地图优化 不在当前帧的点（看不见的点） 删除 匹配次数不高的 点   删除 视角过大删除
 */


#include <boost/timer.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "myslam/config.h"
#include "myslam/visual_odometry.h"
#include "myslam/g2o_types.h"

namespace myslam
{
    VisualOdometry::VisualOdometry() :
    state_(INITIALIZING), ref_(nullptr), curr_(nullptr), map_(new Map), num_lost_(0), num_inliers_(0), matcher_flann_(new cv::flann::LshIndexParams(5,10,2))
    {
        num_of_features_    = Config::get<int>("number_of_features");
        scale_factor_       = Config::get<double>("scale_factor");
        level_pyramid_      = Config::get<int>("level_pyramid");
        match_ratio_        = Config::get<float>("match_ratio");
        max_num_lost_       = Config::get<float>("max_num_lost");
        min_inliers_        = Config::get<int>("min_inliers");
        key_frame_min_rot_  = Config::get<double>("keyframe_rotation");
        key_frame_min_trans_ = Config::get<double>("keyframe_translation");
        map_point_erase_ratio_ = Config::get<double>("map_point_erase_ratio");
        orb_ = cv::ORB::create(num_of_features_, scale_factor_, level_pyramid_);
    }


    VisualOdometry::~VisualOdometry()
    {

    }

    bool VisualOdometry::addFrame(myslam::Frame::Ptr frame)
    {
        switch(state_)
        {
            case INITIALIZING:
            {
                state_ = OK;
                ref_ = curr_ = frame;
                extractKeyPoints();
                computeDescriptors();
                addKeyFrame();
                break;
            }
            case OK:
            {
                curr_ = frame;
                curr_->T_c_w_ = ref_->T_c_w_;
                extractKeyPoints();
                computeDescriptors();
                featureMatching();
                poseEstimationPnP();
                if (true == checkEstimatedPose())
                {
                    curr_->T_c_w_ = T_c_w_estimated_;
                    optimizeMap();
                    num_lost_ = 0;
                    if (true == checkKeyFrame())
                    {
                        addKeyFrame();
                    }
                }
                else
                {
                    num_lost_++;
                    if (num_lost_ > max_num_lost_)
                    {
                        state_ = LOST;
                    }
                    return false;
                }

                break;
            }
            case LOST:
            {
                cout << "vo has lost." << endl;
                break;
            }
        }

        return true;
    }

    void VisualOdometry::extractKeyPoints()
    {
        boost::timer timer;
        orb_->detect(curr_->color_, keypoints_curr_);
        cout << "extract keypoints cost time: " << timer.elapsed() << endl;
    }

    void VisualOdometry::computeDescriptors()
    {
        boost::timer timer;
        orb_->compute(curr_->color_, keypoints_curr_, descriptors_curr_);
        cout << "descriptor computation cost time: " << timer.elapsed() << endl;
    }

    // matching between feature points and the candidates in map
    void VisualOdometry::featureMatching()
    {
        boost::timer timer;
        vector<cv::DMatch> matches;
        // select the candidates in map
        Mat desp_map;
        vector<MapPoint::Ptr> candidate;
        for (auto& allpoints : map_->map_points_)   //& 作用引用，即是别名
        {
            MapPoint::Ptr& p = allpoints.second; // second is a copy of the second object, the first object is key? map_points_.insert(make_pair(map_point->id_, map_point));
            // check if p in curr frame image
            if (curr_->isInframe(p->pos_))
            {
                // add to candidate
                p->visible_times_++;     // the number of this mappoint that being visible in current frame
                candidate.push_back(p);
                desp_map.push_back(p->descriptor_); // store this mappoint's descriptor
            }
        }

        matcher_flann_.match(desp_map, descriptors_curr_, matches); // 当前图像的描述子与存储的地图描述子匹配，匹配对放在matches
        // matches每一个匹配对里边说明匹配信息中来自desp_map和descriptors_curr中的一对描述子，描述子是对关键点周边像素的描述

        // select the best matches
        float min_dis = std::min_element(
                matches.begin(), matches.end(),
                [](const cv::DMatch& m1, const cv::DMatch& m2)
        {
            return m1.distance < m2.distance;
        })->distance;

        match_3dpts_.clear();
        match_2dkp_index_.clear();

        for (cv::DMatch& m:matches)
        {
            if (m.distance < max<float>(min_dis*match_ratio_, 30.0))
            {
                match_3dpts_.push_back(candidate[m.queryIdx]);  // selected mappoint from map_
                match_2dkp_index_.push_back(m.trainIdx); // corresponding matched keypoints in current frame
            }
        }

        cout << "good matches: " << match_3dpts_.size() << endl;
        cout << "match cost time: " << timer.elapsed() << endl;
    }

    // 对于代码，首先从整体上分模块去分析，搞清楚分模块之间的联系，然后再逐个分析单个模块的功能代码，以此类推

    void VisualOdometry::poseEstimationPnP()
    {
        // construct the 3d 2d observations
        vector<cv::Point3f> pts3d;
        vector<cv::Point2f> pts2d;

        for (int index:match_2dkp_index_)
        {
            pts2d.push_back(keypoints_curr_[index].pt);
        }

        for (MapPoint::Ptr pt:match_3dpts_)
        {
            pts3d.push_back(pt->getPositionCV()); // pts3d 与match_3dpts_的关系，pts3d的元素类型是Point3f， match_3dpts元素是vector3d
        }

        Mat K = (cv::Mat_<double>(3,3) <<
                ref_->camera_->fx_, 0, ref_->camera_->cx_,
        0, ref_->camera_->fy_, ref_->camera_->cy_,
        0, 0, 1);

        Mat rvec, tvec, inliers;
        cv::solvePnPRansac(pts3d, pts2d, K, Mat(), rvec, tvec, false, 100, 4.0, 0.99, inliers); // inliers？
        num_inliers_ = inliers.rows;
        cout << "pnp inliers: " << num_inliers_ << endl;
        T_c_w_estimated_ = SE3(
                SO3(rvec.at<double>(0,0), rvec.at<double>(1,0), rvec.at<double>(2,0)),
                Vector3d(tvec.at<double>(0,0), tvec.at<double>(1,0), tvec.at<double>(2,0))
        );

        // using bundle adjustment to optimize the pose
        // _measurement is pts2d, 待优化变量是T_c_w_estimated一个节点，所以为一元边，误差类型为2,2d
        typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 2>> Block;
        Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();
        Block* solver_ptr = new Block(linearSolver);
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm(solver);

        // define an vertex and add it
        g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
        pose->setId(0);
        pose->setEstimate(g2o::SE3Quat(
                T_c_w_estimated_.rotation_matrix(), T_c_w_estimated_.translation()
                ));
        optimizer.addVertex(pose);

        // edges
        for (int i = 0; i < inliers.rows; ++i)
        {
            int index = inliers.at<int>(i,0);   // 匹配上的点序号
            EdgeProjectXYZ2UVPoseOnly* edge = new EdgeProjectXYZ2UVPoseOnly();
            edge->setId(i);
            edge->setVertex(0,pose);
            edge->camera_ = curr_->camera_.get();   // fucntion of shared_ptr that can return smart pointer
            edge->point_ = Vector3d(pts3d[index].x, pts3d[index].y, pts3d[index].z);
            edge->setMeasurement(Vector2d(pts2d[index].x, pts2d[index].y) );
            edge->setInformation(Eigen::Matrix2d::Identity());
            optimizer.addEdge(edge);
            match_3dpts_[index]->matched_times_++;  // line173 确定了内点,匹配良好的mappoint个数
        }

        optimizer.initializeOptimization();
        optimizer.optimize(10);

        T_c_w_estimated_ = SE3(
                pose->estimate().rotation(),
                pose->estimate().translation()
                );

        cout << "T_c_w_estimated_: "<< endl<<T_c_w_estimated_.matrix()<<endl;
    }

    bool VisualOdometry::checkEstimatedPose()
    {
        // check if the estimated pose is good
        if (num_inliers_ < min_inliers_)
        {
            cout << "reject because inlier is too small: " << num_inliers_ << endl;
            return false;
        }

        // if the motion is too large, it is probably wrong
        SE3 T_r_c = ref_->T_c_w_ * T_c_w_estimated_.inverse();
        // T_c_w_ :world-->camera; T_c_w_estimated_:world-->camera. 左乘
        Sophus::Vector6d d = T_r_c.log();
        if (d.norm() > 5.0)
        {
            cout << "reject because motion is too large: " << d.norm() << endl;
            return false;
        }

        return true;
    }

    bool VisualOdometry::checkKeyFrame()
    {
        SE3 T_r_c =ref_->T_c_w_ * T_c_w_estimated_.inverse();
        Sophus::Vector6d d = T_r_c.log();
        Vector3d trans = d.head<3>();
        Vector3d rot = d.tail<3>();
        if (rot.norm() > key_frame_min_rot_ || trans.norm() > key_frame_min_trans_)
        {
            return true;
        }

        return false;
    }

    // 在提取第一帧的特征点之后，将第一帧的所有特征点全部放入地图中,这样才有起始的地图，给后续的帧提供feature matching，然后不断更新map_中的mappoints_,维持一个局域的地图
    void VisualOdometry::addKeyFrame()
    {
        if (map_->keyframes_.empty())
        {
            for (size_t i = 0; i < keypoints_curr_.size(); i++)
            {
                double d = curr_->findDepth(keypoints_curr_[i]);
                if (d < 0)
                    continue;
                Vector3d p_world = ref_->camera_->pixel2world(
                        Vector2d (keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y), curr_->T_c_w_, d
                );

                Vector3d n = p_world - ref_->getCamCenter();
                n.normalize();
                MapPoint::Ptr map_point = MapPoint::createMapPoint( p_world, n, descriptors_curr_.row(i).clone(), curr_.get() );
                map_->insertMapPoint(map_point);    // 创建mappoint设置其id,每创建一次id递增一次，id代表了创建的mappoint个数，并添加到map_中
            }
        }

        map_->insertKeyFrame(curr_);    // 添加关键帧
        ref_ = curr_;
    }

    void VisualOdometry::addMapPoints()
    {
        // add new map points into map
        vector<bool> matched(keypoints_curr_.size(), false);    // keypoints_curr_.size()为大小，初值为false
        //这里循环，应该是不断取vector match_2dkp_index_中存储的keypoints的匹配上的关键点的索引
        for (int index:match_2dkp_index_)
        {
            matched[index] = true;  // 将keypoints_curr_中匹配成功的那部分关键点序号对应的matched设置为true，
        }

        for (int i = 0; i < keypoints_curr_.size(); ++i)
        {
            // 没有匹配到的点，说明是之前地图中没有的，当前地图中新出现的特征点；
            //所以匹配不上,那么就加入到地图中，给后续的图片匹配使用；
                    // 已经匹配到的点继续保留在里边呢，是怎么来的（是在之前的帧数中添加的第一帧 或者前面帧已经添加过了）
            if (matched[i] == true)
                continue;
            double d = ref_->findDepth(keypoints_curr_[i]);
            if (d < 0)
                continue;
            Vector3d p_world = ref_->camera_->pixel2world(
                    Vector2d(keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y),
            curr_->T_c_w_, d);  // 求解curr_中像素对应的世界坐标系空间点坐标
            Vector3d n = p_world - ref_->getCamCenter();
            n.normalize();
            MapPoint::Ptr map_point = MapPoint::createMapPoint(p_world, n, descriptors_curr_.row(i).clone(), curr_.get()
                    );  // get 函数是智能指针的成员函数，所以用成员操作符.
            map_->insertMapPoint(map_point); // 添加新的mappoint
        }
    }

    void VisualOdometry::optimizeMap()
    {
        // remove the hardly seen and no visible points  Camera lense adjustments
        //-light intensity
        //-some adjustment on colon surface color and wetness
        for (auto iter = map_->map_points_.begin(); iter != map_->map_points_.end(); )
        {
            if (!curr_->isInframe(iter->second->pos_))
            {
                iter = map_->map_points_.erase(iter);   // 删除掉旧的，后面一个补上来？
                continue;   //把map_中不在当前frame中的mappoints清除掉
            }

            // 如果map_->map_points_在当前的frame中,但是该点的成功匹配
            // 次数相对于被观察到次数（出现在当前帧次数）比较低，说明贡献不大
            float match_ratio = float(iter->second->matched_times_) / iter->second->visible_times_;
            if (match_ratio < map_point_erase_ratio_)
            {
                iter = map_->map_points_.erase(iter);
                continue;
            }

            double angle = getViewAngle(curr_, iter->second);
            if (angle > M_PI /6.)
            {
                iter = map_->map_points_.erase(iter);
                continue;
            }

            if (iter->second->good_ == false)   // 什么时候设置的值呢
            {
                // TODO try triangulate this map point
            }
            iter++;
        }

        if (match_2dkp_index_.size() < 100)
            addMapPoints(); //匹配点对小于 100个  两幅图像重合度过小 添加点到新地图
        if (map_->map_points_.size() > 1000)
        {
            // map is too large, remove some one
            map_point_erase_ratio_ += 0.05; //增大 匹配率 门槛限制，对应336行代码，提高比例，这样更多的点符合删除要求
        }
        else
        {
            map_point_erase_ratio_ = 0.1;
        }

        cout << "map points: " << map_->map_points_.size() << endl;
    }

    double VisualOdometry::getViewAngle(myslam::Frame::Ptr frame, myslam::MapPoint::Ptr point)
    {
        Vector3d n = point->pos_ - frame->getCamCenter();
        n.normalize();
        return acos(n.transpose() * point->norm_);  // 理解？ a * b = |a||b|cos<a,b>
        // 因为normalize所以|a|= |b| = 1, 因此 a * b = cos<a,b>, 即 acos(a*b) = <a,b> 向量之间的夹角
        // 当前帧的相机光心世界坐标与map中的一个mappoint(地图中的一个世界坐标系中的三维点)的视角差
        // 对地图中的每一个landmark都进行视角差测试，视角差过大，比如说大于30°，就说明当前帧与这一个mappoint已经相聚较远了；
        // 因而可以删除这个地图点了
    }



}