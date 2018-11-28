//
// Created by nuc on 18-7-5.
//
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <chrono>

using namespace std;
using namespace cv;

void find_feature_matches(
        const Mat& img_1, const Mat& img_2,
        std::vector<KeyPoint>& keypoints_1,
        std::vector<KeyPoint>& keypoints_2,
        std::vector<DMatch>& matches );

Point2f pixel2cam(const Point2d& p, const Mat& K);

void bundleAdjustment(
        const vector<Point3f> points_3d,
        const vector<Point2f> points_2d,
        const Mat& K,
        Mat& R, Mat& t
        );

int main(int argc, char** argv)
{
    if (argc != 5)
    {
        cout << "usage: pose_estimation_3d2d img1 img2 depth1 depth2" << endl;
        return 1;
    }

    Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    vector<KeyPoint> keypoints_1;
    vector<KeyPoint> keypoints_2;
    vector<DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "find " << matches.size() << " pairs of matched features" << endl;

    //-- build 3D points from img_1 and depth_1
    Mat d1 = imread(argv[3], CV_LOAD_IMAGE_UNCHANGED);
    Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
    vector<Point3f> pts_3d;
    vector<Point2f> pts_2d;

    for (DMatch m:matches)
    {
        ushort d = d1.ptr<unsigned short>( int (keypoints_1[m.queryIdx].pt.y) ) [ int (keypoints_1[m.queryIdx].pt.x) ];
        if ( d == 0 )
            continue;
        float dd = d / 1000.0;
        Point2f p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        pts_3d.push_back( Point3f( p1.x * dd, p1.y * dd, dd ) );
        pts_2d.push_back(keypoints_2[m.trainIdx].pt);
    }

    cout << "3d-2d pairs: " << pts_3d.size() << endl;
    Mat r, t;
    //solvePnP( pts_3d, pts_2d, K, Mat(), r, t, false,);
    solvePnP(pts_3d, pts_2d, K, Mat(), r, t, false, cv::SOLVEPNP_EPNP);
    Mat R;
    cv::Rodrigues(r , R);
    cout << "R= " << endl << R << endl;
    cout << "t= " << endl << t << endl;

    cout << "calling bundle adjustment" << endl;
    bundleAdjustment(pts_3d, pts_2d, K, R, t);
}


void find_feature_matches(
        const Mat& img_1, const Mat& img_2,
        std::vector<KeyPoint>& keypoints_1,
        std::vector<KeyPoint>& keypoints_2,
        std::vector<DMatch>& matches )
{
    Mat descriptors_1, descriptors_2;

    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");


    //--第一步： 检测Oriented FAST角点位置
    detector->detect( img_1, keypoints_1, descriptors_1 );
    detector->detect( img_2, keypoints_2, descriptors_2 );

    //--第二步： 根据角点位置计算BRIEF描述子
    descriptor->compute( img_1, keypoints_1, descriptors_1 );
    descriptor->compute( img_2, keypoints_2, descriptors_2 );

    //--第三步： 对两幅图像中的BRIEF描述子进行匹配，使用Hamming距离
    vector<DMatch> match;
    matcher->match(descriptors_1, descriptors_2, match);

    double min_dist = 10000, max_dist = 0;
    for (int i = 0; i < match.size(); ++i)
    {
        // 将每一对的匹配点距离进行对比，找出最大与最小； match.size 与descriptors_1.rows
        // 然后留下距离小于最小距离二倍的匹配对，由于最小距离可能比较小，可能需要设置一个经验值，如果最小距离太小，就用这个值，否则若最小距离大于经验值，就用最小距离作为衡量标准。
        double dist = match[i].distance;
        if (dist > max_dist)
            max_dist = dist;
        if (dist < min_dist)
            min_dist = dist;
    }

    printf("-- Max dist: %f \n", max_dist);
    printf("-- Min dist: %f \n", min_dist);

    for (int j = 0; j < descriptors_1.rows; ++j)
    {
        if ( match[j].distance < max(min_dist, 30.0) )
        {
            matches.push_back( match[j] );
        }
    }
}

Point2f pixel2cam(const Point2d& p, const Mat& K)
{
    return Point2f( (p.x - K.at<double> ( 0,2 ) ) / K.at<double> (0,0), ( p.y - K.at<double> ( 1,2 ) ) / K.at<double> ( 1,1 ) );
}

void bundleAdjustment(
        const vector<Point3f> points_3d,
        const vector<Point2f> points_2d,
        const Mat& K,
        Mat& R, Mat& t
)
{
    //-- 初始化g2o
    //-- 矩阵块：每个误差项优化变量维度为6（第二个相机的位姿se(3)）， 误差值维度为3（每个3D点在第二个相机中的投影）
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<6,3> > Block;
    //--线性方程求解器：稠密的增量方程
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverCSparse<Block::PoseMatrixType> ();
    Block* solver_ptr = new Block( linearSolver ); // 设置后矩阵块的优化变量、误差值维度，以及求解器的类型后，创建一个矩阵块求解器，接下来设置优化方法
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( solver_ptr);
    g2o::SparseOptimizer optimizer; // 图模型
    optimizer.setAlgorithm( solver ); // 设置求解器

    //-- 向图中添加顶点
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    Eigen::Matrix3d R_mat;
    R_mat <<
          R.at<double> ( 0,0 ), R.at<double> ( 0,1 ), R.at<double> ( 0,2 ),
          R.at<double> ( 1,0 ), R.at<double> ( 1,1 ), R.at<double> ( 2,1 ),
          R.at<double> ( 2,0 ), R.at<double> ( 2,1 ), R.at<double> ( 2,2 );
    pose->setId( 0 );
    pose->setEstimate( g2o::SE3Quat( R_mat, Eigen::Vector3d ( t.at<double> ( 0,0 ), t.at<double> ( 1,0 ), t.at<double> ( 2,0 ) ) ) );
    optimizer.addVertex( pose );

    int index = 1;
    for ( const Point3f p:points_3d ) // landmarks
    {
        g2o::VertexSBAPointXYZ* point = new g2o::VertexSBAPointXYZ();
        point->setId( index++ );
        point->setEstimate( Eigen::Vector3d ( p.x, p.y, p.z ) );
        point->setMarginalized( true );
        optimizer.addVertex( point );
    }

    // parameter: camera intrinsics
    g2o::CameraParameters* camera = new g2o::CameraParameters(
            K.at<double> ( 0,0 ), Eigen::Vector2d ( K.at<double> ( 0,2 ), K.at<double> ( 1,2 ) ), 0
    );
    camera->setId( 0 );
    optimizer.addParameter( camera );

    index = 1;
    for ( const Point2f p:points_2d )
    {
        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
        edge->setId( index );
        edge->setVertex( 0, dynamic_cast<g2o::VertexSBAPointXYZ*>( optimizer.vertex( index ) ) );  //向边添加0类型节点，参考帧中的空间3D点作为当前帧的优化变量之一
        edge->setVertex( 1, pose ); // 当前帧的位姿作为节点，即优化变量
        edge->setMeasurement( Eigen::Vector2d ( p.x, p.y ) );
        edge->setParameterId( 0,0 );
        edge->setInformation( Eigen::Matrix2d::Identity() );
        optimizer.addEdge( edge );
        index++;
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose( true );
    optimizer.initializeOptimization();
    optimizer.optimize( 100 );
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> ( t2-t1 );
    cout <<"optimization costs time: " << time_used.count() << " seconds." <<endl;
    cout<<endl<<"after optimization: " << endl;
    cout <<"T="<<endl<<Eigen::Isometry3d( pose->estimate() ).matrix() << endl;

}
