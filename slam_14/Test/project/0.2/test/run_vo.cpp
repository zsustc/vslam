//
// Created by zbox on 18-8-29.
//
#include <iostream>
#include <fstream>
#include <boost/timer.hpp>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/viz.hpp>


#include "myslam/config.h"
#include "myslam/visual_odometry.h"
using namespace std;
using namespace myslam;

int main(int argc, char** argv)
{
    if (2 != argc)
    {
        cout << "usage: run_vo parameter_file" << endl;
        return 1;
    }

    myslam::Config::setParameterFile(argv[1]);
    //myslam::VisualOdometry::Ptr vo( new myslam::VisualOdometry);

    myslam::VisualOdometry::Ptr vo(new myslam::VisualOdometry);

    string dataset_dir = myslam::Config::get<string>("dataset_dir");
    cout << "dataset: " << dataset_dir << endl;
    ifstream fin(dataset_dir+"/associate.txt");
    if (!fin)
    {
        cout << "Please generate the associate file called associate.txt" << endl;
        return 1;
    }

    vector<string> rgb_files, depth_files;
    vector<double> rgb_times, depth_times;

    // 读取并存储色彩图和灰度图
    while(!fin.eof())
    {
        string rgb_time, rgb_file, depth_time, depth_file;
        fin >>rgb_time>>rgb_file>>depth_time>>depth_file;
        rgb_times.push_back(atof(rgb_time.c_str()));
        depth_times.push_back(atof(depth_time.c_str()));
        rgb_files.push_back(dataset_dir+"/"+rgb_file);
        depth_files.push_back(dataset_dir+"/"+depth_file);

        if (fin.good() == false)
            break;
    }

    myslam::Camera::Ptr camera(new myslam::Camera);

    // visualization
    cv::viz::Viz3d vis("Visual Odometry");
    cv::viz::WCoordinateSystem world_coor(1.0), camera_coor(0.5);
    cv::Point3d cam_pos(0, -1.0, -1.0), cam_focal_point(0,0,0),cam_y_dir(0,1,0);
    cv::Affine3d cam_pose = cv::viz::makeCameraPose(cam_pos, cam_focal_point, cam_y_dir);
    vis.setViewerPose(cam_pose);

    world_coor.setRenderingProperty(cv::viz::LINE_WIDTH, 2.0);
    camera_coor.setRenderingProperty(cv::viz::LINE_WIDTH, 1.0);
    vis.showWidget("World", world_coor);
    vis.showWidget("Camera", camera_coor);

    cout << "real total " <<rgb_files.size() <<" entries"<<endl;
    for (int i = 0; i < rgb_files.size(); ++i)
    {
        Mat color = cv::imread(rgb_files[i]);
        Mat depth = cv::imread(depth_files[i], -1);
        if (nullptr == color.data || nullptr == depth.data)
            break;
        myslam::Frame::Ptr pFrame = myslam::Frame::createFrame();
        pFrame->camera_ = camera;
        pFrame->color_ = color;
        pFrame->depth_ = depth;
        pFrame->time_stamp_ = rgb_times[i];

        boost::timer timer;
        vo->addFrame(pFrame);
        cout << "VO costs time: " << timer.elapsed()<< endl;

        if (vo->state_ == myslam::VisualOdometry::LOST)
        {
            break;
        }

        // 第一帧图像的位姿在哪里设置的呢，E 单位矩阵？对，已经gdb调试出是单位矩阵
        SE3 Tcw = pFrame->T_c_w_.inverse();  //  T_c_w, transform from world to camera, Tcw camera to world

        // show the map and the camera pose
        cv::Affine3d M(
                cv::Affine3d::Mat3(Tcw.rotation_matrix()(0,0), Tcw.rotation_matrix()(0,1),Tcw.rotation_matrix()(0,2),
                                   Tcw.rotation_matrix()(1,0),Tcw.rotation_matrix()(1,1),Tcw.rotation_matrix()(1,2),
                                   Tcw.rotation_matrix()(2,0),Tcw.rotation_matrix()(2,1),Tcw.rotation_matrix()(2,2)
                ),
                cv::Affine3d::Vec3(
                        Tcw.translation()(0,0), Tcw.translation()(1,0),Tcw.translation()(2,0)
                )
        );

        cv::imshow("image", color);
        cv::waitKey(1 );
        vis.setWidgetPose("Camera", M); //根据M,将图像color按照其在世界坐标系中的位姿显示出来，其实就是相机的位姿
        vis.spinOnce(1,false);  // ？
    }

    return 0;

}
