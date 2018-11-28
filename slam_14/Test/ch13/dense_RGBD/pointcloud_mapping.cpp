//
// Created by zbox on 18-10-27.
//

#include <iostream>
#include <fstream>

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/StdVector>
using namespace Eigen;

#include <boost/format.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
//#include <pcl/visualization/pcl_visualizer.h>

int main(int argc, char** argv)
{
    vector<cv::Mat> colorImgs, depthImgs;
    vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> poses;

    ifstream fin("./pose.txt");
    if (!fin)
    {
        cerr << "cannot find pose file" << endl;
        return 1;
    }

    // 读取RGBD图片信息，读取位姿信息
    for (int i = 0; i < 5; ++i)
    {
        boost::format fmt("./%s/%d.%s");
        colorImgs.push_back( cv::imread( (fmt%"color"%(i+1)%"png").str() ));
        depthImgs.push_back(cv::imread( (fmt%"depth"%(i+1)%"pgm").str(), -1) );

        double data[7] = {0};
        for (auto& d:data)
            fin>>d;


//        for (int j = 0; j < 7; ++j)
//        {
//            fin >> data[j];
//        }

        Eigen::Quaterniond q(data[6], data[3], data[4], data[5]);
        Eigen::Isometry3d T(q);
        T.pretranslate(Eigen::Vector3d(data[0], data[1], data[2]));
        poses.push_back(T);
    }

    double cx = 325.5;
    double cy = 253.5;
    double fx = 518.0;
    double fy = 519.0;
    double depthScale = 1000.0;

    cout << "正在将图像转换为点云" << endl;
    typedef pcl::PointXYZRGB PointT;
    typedef pcl::PointCloud<PointT> PointCloud;

    PointCloud::Ptr pointCloud(new PointCloud);
    for (int j = 0; j < 5; ++j)
    {
        PointCloud::Ptr current(new PointCloud);
        cout << "转换图像中："<< j+1 << endl;
        cv::Mat color = colorImgs[j];
        cv::Mat depth = depthImgs[j];
        Eigen::Isometry3d T = poses[j];
        for (int v = 0; v < color.rows; ++v)
        {
            for (int u = 0; u < color.cols; ++u)
            {
                unsigned int d = depth.ptr<unsigned short>(v)[u];
                if (0 == d)
                    continue;   //没有测量到深度
                if (d >= 7000)  //深度太大时不稳定，去掉
                    continue;
                Vector3d point;
                point[2] = double(d) / depthScale;
                point[0] = point[2] * (u - cx) / fx;
                point[1] = point[2] * (v - cy) / fy;
                Eigen::Vector3d pointWolrd = T * point;

                PointT p;
                p.x = pointWolrd[0];
                p.y = pointWolrd[1];
                p.z = pointWolrd[2];
                p.b = color.data[v * color.step + u*color.channels() + 0];
                p.g = color.data[v * color.step + u * color.channels() + 1];
                p.r = color.data[v * color.step + u * color.channels() + 2];
                current->points.push_back(p);
            }
        }

        // depth filter and statistical removal
        PointCloud::Ptr tmp(new PointCloud);
        pcl::StatisticalOutlierRemoval<PointT> statistical_filter;
        statistical_filter.setMeanK(50);
        statistical_filter.setStddevMulThresh(1.0);
        statistical_filter.setInputCloud(current);
        statistical_filter.filter(*tmp);
        (*pointCloud) += *tmp;
    }

    pointCloud->is_dense = false;
    cout << "点云共有"<<pointCloud->size()<<"个点"<<endl;

    // voxel filter
    pcl::VoxelGrid<PointT> voxel_filter;
    voxel_filter.setLeafSize(0.01, 0.01, 0.01);
    PointCloud::Ptr tmp(new PointCloud);
    voxel_filter.setInputCloud(pointCloud);
    voxel_filter.filter(*tmp);
    tmp->swap(*pointCloud);

    cout<<"滤波之后，点云共有"<<pointCloud->size()<<"个点."<<endl;

    pcl::io::savePCDFileBinary("map.pcd", *pointCloud);
    return 0;




}