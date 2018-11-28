//
// Created by zbox on 18-10-29.
//

#include <iostream>
#include <fstream>

using namespace std;
// eigen
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
using namespace Eigen;

// opencv
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
using namespace cv;

// pcl
#include <pcl/io/pcd_io.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>

// for format
#include <boost/format.hpp>


// for octomap
#include <octomap/octomap.h>


int main(int argc, char** argv)
{
    // load color images, depth images and poses
    ifstream fin("./data/pose.txt");
    if(!fin)
    {
        cerr << "cannot find pose file" << endl;
    }

    boost::format fmt("./data/%s/%d.%s");
    std::vector<cv::Mat> color_images, depth_images;
    std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> poses;

    for (int i = 0; i < 5; ++i)
    {
        color_images.push_back( cv::imread( (fmt%"color"%(i+1)%"png").str()) );
        depth_images.push_back( cv::imread( (fmt%"depth"%(i+1)%"pgm").str(),-1) );

        double data[7] = {0};
        for (int j = 0; j < 7; ++j)
        {
            fin>>data[j];
        }

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

    cout << "正在将图像转换为 Octomap..." << endl;
    octomap::OcTree tree(0.05);

    for (int k = 0; k < 5; ++k)
    {
        cv::Mat color = color_images[k];
        cv::Mat depth = depth_images[k];
        Eigen::Isometry3d T = poses[k];
        octomap::Pointcloud cloud;

        for (int v = 0; v < color.rows; ++v)
        {
            for (int u = 0; u < color.cols; ++u)
            {
                unsigned int d = depth.ptr<unsigned short>(v)[u];
                if (0 == d) // no depth
                    continue;
                if (7000 <= d) // depth is too ;large
                    continue;
                Eigen::Vector3d point;
                point[2] = double(d) / depthScale;
                point[0] = (u - cx) * point[2] / fx;
                point[1] = (v - cy) * point[2] / fy;

                Eigen::Vector3d pointWorld = T*point;
                cloud.push_back( pointWorld[0], pointWorld[1], pointWorld[2] );
            }
        }

        tree.insertPointCloud(cloud, octomap::point3d(T(0,3), T(1,3), T(2,3) ) );
    }

    tree.updateInnerOccupancy();
    cout << "saving octomap ... " << endl;
    tree.writeBinary("octomap.bt");
    return 0;

}