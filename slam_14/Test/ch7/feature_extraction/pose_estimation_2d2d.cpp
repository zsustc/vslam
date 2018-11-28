//
// Created by nuc on 18-6-26.
//
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;

void find_feature_matches(
        const Mat& img_1, const Mat& img_2,
        std::vector<KeyPoint>& keypoints_1,
        std::vector<KeyPoint>& keypoints_2,
        std::vector<DMatch>& matches);

void pose_estimation_2d2d(
        std::vector<KeyPoint> keypoints_1,
        std::vector<KeyPoint> keypoints_2,
        std::vector<DMatch> matches,
        Mat& R, Mat& t);

Point2f pixel2cam ( const Point2d& p, const Mat& K );

void triangulation(
        const vector<KeyPoint>& keypoint_1,
        const vector<KeyPoint>& keypoint_2,
        const std::vector<DMatch> matches,
        const Mat& R, const Mat& t,
        vector<Point3d>& points);

int main(int argc, char** argv)
{
    if(argc != 4)
    {
        cout << "usage: feature_extraction img1 img2" << endl;
        return 1;
    }

    Mat img1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    std::vector<KeyPoint> keypoints_1;
    std::vector<KeyPoint> keypoints_2;
    std::vector<DMatch> matches;

    find_feature_matches(img1, img2, keypoints_1, keypoints_2, matches);
    cout << "一共找到了" << matches.size() << "组匹配点" << endl;

    Mat R,t;
    pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);

    // validating E = t^R*scale
    Mat t_x = (Mat_ <double> (3, 3) <<
            0,  -t.at<double>(2,0),  t.at<double>(1,0),
            t.at<double>(2,0),  0,  -t.at<double>(0,0),
            -t.at<double>(1,0), t.at<double>(0,0),   0);

    cout << "t^R= " << t_x * R << endl;

    // validating epipolar constraint
    Mat K = (Mat_<double> (3,3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    for (DMatch m:matches)
    {
        Point2d pt1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        Mat y1 = (Mat_ <double>(3,1) << (pt1.x, pt1.y, 1);
        Point2d pt2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
        Mat y2 = (Mat_ <double>(3,1) << (pt2.x, pt2.y, 1);
        Mat d = y2.t() * t_x * R * y1;
        cout << "epipolar constraint = " << d << endl;
    }

    // triangulation for measuring scale
    vector<Point3d> points;
    triangulation(keypoints_1, keypoints_2, matches, R, t, points);

    //--验证三角化点与特征点的重投影关系
    // Mat K = (Mat_<double>(3,3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0,0,1);
    for (int i = 0; i < matches.size(); ++i)
    {
        Point2d pt1_cam = pixel2cam(keypoints_1[matches[i].queryIdx].pt, K);
        Point2d pt1_cam_3d(
                points[i].x/points[i].z,
                points[i].y/points[i].z
        );

        cout << "point in the first camera frame: " << pt1_cam << endl;
        cout << "point projected from 3D " << pt1_cam_3d << ",d=" << points[i].z << endl;

        // the second image
        Point2f  pt2_cam = pixel2cam(keypoints_2[matches[i].trainIdx].pt, K);
        Mat pt2_trans = R*(Mat_<double>(3,1) << points[i].x, points[i].y, points[i].z) + t;
        pt2_trans /= pt2_trans.at<double>(2,0);
        cout << "point in the second camera frame: " << pt2_cam << endl;
        cout << "point projected from second frame " << pt2_trans.t() << endl;
        cout << endl;
    }

    /*************************************
    pose_estimation_3d2d.cpp
     结果不准确问题可能出现在特征提取环节，特征匹配出了问题
     keypoints and descriptors
    *************************************/
    Mat d1 = imread(argv[3], CV_LOAD_IMAGE_UNCHANGED);
    vector<Point3f> pts_3d;
    vector<Point2f> pts_2d;

    for(DMatch m:matches)
    {
        ushort d = d1.ptr<unsigned short>( int(keypoints_1[m.queryIdx].pt.y) )[int(keypoints_1[m.queryIdx].pt.x)];
        if (d == 0)
            continue;
        float dd = d / 1000.0;
        Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        pts_3d.push_back(Point3f(p1.x * dd, p1.y * dd, dd));
        pts_2d.push_back(keypoints_2[m.trainIdx].pt);
    }

    cout << "3d-2d pairs: " << pts_3d.size() << endl;
    Mat r1,t1;
    // 调用opencv的PnP求解，可选择EPNP，DLS等方法
    solvePnP(pts_3d, pts_2d, K, Mat(), r1, t1, false, cv::SOLVEPNP_EPNP);
    Mat R1;
    cv::Rodrigues(r1, R1);

    cout << "R = " << endl << R1 << endl;
    cout << "t = " << t1 << endl;

    return 0;
}

void find_feature_matches(
        const Mat& img_1, const Mat& img_2,
        std::vector<KeyPoint>& keypoints_1,
        std::vector<KeyPoint>& keypoints_2,
        std::vector<DMatch>& matches)
{
    //-- 初始化
    Mat descriptors_1, descriptors_2;
    // used in OpenCV3
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();

    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect ( img_1,keypoints_1 );
    detector->detect ( img_2,keypoints_2 );

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute ( img_1, keypoints_1, descriptors_1 );
    descriptor->compute ( img_2, keypoints_2, descriptors_2 );

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> match;
    //BFMatcher matcher ( NORM_HAMMING );
    matcher->match ( descriptors_1, descriptors_2, match );

    double min_dist = 10000, max_dist =0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = match[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( match[i].distance <= max ( 2*min_dist, 30.0 ) )
        {
            matches.push_back ( match[i] );
        }
    }
}

void pose_estimation_2d2d(
        std::vector<KeyPoint> keypoints_1,
        std::vector<KeyPoint> keypoints_2,
        std::vector<DMatch> matches,
        Mat& R, Mat& t)
{
    // camera intrinsic parameters, TUM Freburg2
    Mat K =(Mat_<double>(3,3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    // 把匹配点转换为 vector<Point2f> 的形式
     vector<Point2f> points1;
     vector<Point2f> points2;

    for (int i = 0; i < (int) matches.size(); ++i)
    {
        points1.push_back(keypoints_1[matches[i].queryIdx].pt);
        //points2.push_back(keypoints_2[matches[i].queryIdx].pt);
        points2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }

    //calculating fundamental matrix
    Mat fundamental_matrix;
    fundamental_matrix = findFundamentalMat(points1, points2, CV_FM_8POINT);
    cout << "fundamental_matrix is " << endl << fundamental_matrix << endl;

    Point2d principal_point (325.1, 249.7);
    int focal_length = 521;
    Mat essential_matrix;
    essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point, RANSAC);
    cout << "essential_matrix is " << endl << essential_matrix << endl;

    //--计算homography matrix
    Mat homography_matrix = findHomography(points1, points2, RANSAC, 3, noArray(), 2000, 0.99);
    cout << "homography_matrix is " << homography_matrix << endl;

    //--calculating rotation and translation information
    recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
    cout << "R is " << endl << R << endl;
    cout << "t is " << t << endl;
}

Point2f pixel2cam ( const Point2d& p, const Mat& K )
{
    return Point2f
            (
                    ( p.x - K.at<double> ( 0,2 ) ) / K.at<double> ( 0,0 ),
                    ( p.y - K.at<double> ( 1,2 ) ) / K.at<double> ( 1,1 )
            );
}


void triangulation(
        const vector<KeyPoint>& keypoint_1,
        const vector<KeyPoint>& keypoint_2,
        const std::vector<DMatch> matches,
        const Mat& R, const Mat& t,
        vector<Point3d>& points)
{
    Mat T1 = (Mat_<double> (3,4) <<
            1,0,0,0,
            0,1,0,0,
            0,0,1,0);
    Mat T2 = (Mat_<double>(3,4) <<
            R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0,0),
            R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1,0),
            R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2,0)
    );

    Mat K = (Mat_<double> (3,3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<Point2f> pts_1, pts_2;

    for (DMatch m:matches)
    {
        // 将像素坐标转换到相机坐标
        pts_1.push_back(pixel2cam(keypoint_1[m.queryIdx].pt, K));
        pts_2.push_back(pixel2cam(keypoint_2[m.trainIdx].pt, K));
    }

    Mat pts_4d;
    cv::triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);

    for (int i = 0; i < pts_4d.cols; ++i)
    {
        Mat x = pts_4d.col(i);
        x /= x.at<float>(3,0);

        Point3d p(
                x.at<float>(0,0),
                x.at<float>(1,0),
                x.at<float>(2,0)
        );

        points.push_back(p);
    }

}
