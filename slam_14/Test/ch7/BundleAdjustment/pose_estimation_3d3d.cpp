//
// Created by zbox on 18-8-11.
//
#include <iostream> // input vs output streams

// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

// Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>

// g2o
#include <g2o/core/base_vertex.h> //顶点
#include <g2o/core/base_unary_edge.h> // 一元边
#include <g2o/core/block_solver.h> //矩阵块 分解 求解器 矩阵空间 映射 分解
#include <g2o/core/optimization_algorithm_gauss_newton.h> // 高斯牛顿法
#include <g2o/solvers/eigen/linear_solver_eigen.h>  //矩阵优化
#include <g2o/types/sba/types_six_dof_expmap.h> //定义好的顶点类型和误差 变量更新算法 6维度 例如相机位置姿态
#include <chrono>

using namespace std;    //标准库 命名空间
using namespace cv;     // opencv 命名空间

// 特征提取与匹配，通过对极几何约束计算相机位姿或者结合深度信息计算空间点坐标3的-2d或者3d-3d的pnp/icp/ba方法求解相机位姿
void find_feature_matches(const Mat& img_1, const Mat& img_2,
                          std::vector<KeyPoint>& keypoints_1,
                          std::vector<KeyPoint>& keypoints_2,
                          std::vector<DMatch>& matches);

// 通过相机内参K,将像素坐标转换到相机归一化平面
Point2d pixel2cam(const Point2d& p, const Mat & K);

//通过ICP算法，计算二组匹配3D点之间的位姿变换
void pose_estimation_3d3d(const vector<Point3f>& pts1,
                          const vector<Point3f>& pts2,
                          Mat& R, Mat& t);

// g2o优化 ICP算法求解的R,t作为图优化的初始值，建立顶点和边，进行优化，进一步精确R,t
void bundleAdjustment(const vector<Point3f>& pts1,
                      const vector<Point3f>& pts2,
                      Mat& R, Mat& t);

// 节点为优化变量，边为误差项
// 一元边unary edge ，即只链接一个顶点,表示误差项与一个优化变量有关
// BaseUnaryEdge<3, Eigen::Vector3d, g2o::VertexSE3Expmap> 这里的顶点是相机的位姿，类型为g2o::VertexSE3Expmap，也就是待优化变量；
// 边的维度为3，数据类型为Eigen::Vector3d，是测量到的3D点p与根据R,t计算R^p'+t (p'为观测点）得到的估计点之间的差值；
//
class EdgeProjectXYZRGBDPoseOnly : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;    // 对其符号
    EdgeProjectXYZRGBDPoseOnly(const Eigen::Vector3d& point):_point(point) {}   //通过构造函数，将观测点point直接赋值给_point
    // 误差计算公式
    virtual void  computeError()
    {
        const g2o::VertexSE3Expmap* pose = static_cast<const g2o::VertexSE3Expmap*> (_vertices[0]); //_vertices[0] 0号顶点为位姿，类型强制转换
        _error = _measurement - pose->estimate().map(_point); // measurement is p, point is p'
    }

    //       Vector3d map(const Vector3d & xyz) const
    //      {
    //        return _r*xyz + _t;
    //      }

    virtual void linearizeOplus()
    {
        g2o::VertexSE3Expmap* pose = static_cast<g2o::VertexSE3Expmap*> (_vertices[0]); // 取估计的相机位姿R,t，用来计算观测值p'的变换后估计值R*p'+t
        g2o::SE3Quat T(pose->estimate());   // 取相机位姿，用来计算观测值p'的变换后估计值R*p'+t
        Eigen::Vector3d xyz_trans = T.map(_point);  //公式7.60,计算出观测点在p所在坐标系下的坐标P'
        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double z = xyz_trans[2];

        /* P'=exp(Xi^)p', 其对delta(Xi)的偏导数=[I, -P'^], 3行*6列
         * = [ 1 0 0 0 Z' -Y'; 0 1 0 -Z' 0 ; 0 0 1 Y' -X' 0]
         * 误差项对于位姿Xi的导数等于 -P'
         * */
        _jacobianOplusXi(0,0) = 0;
        _jacobianOplusXi(0,1) = -z;
        _jacobianOplusXi(0,2) = y;
        _jacobianOplusXi(0,3) = -1;
        _jacobianOplusXi(0,4) = 0;
        _jacobianOplusXi(0,5) = 0;

        _jacobianOplusXi(1,0) = z;
        _jacobianOplusXi(1,1) = 0;
        _jacobianOplusXi(1,2) = -x;
        _jacobianOplusXi(1,3) = 0;
        _jacobianOplusXi(1,4) = -1;
        _jacobianOplusXi(1,5) = 0;

        _jacobianOplusXi(2,0) = -y;
        _jacobianOplusXi(2,1) = x;
        _jacobianOplusXi(2,2) = 0;
        _jacobianOplusXi(2,3) = 0;
        _jacobianOplusXi(2,4) = 0;
        _jacobianOplusXi(2,5) = -1;
    }

    bool read(istream& in) {}
    bool write(ostream& out) const {}

protected:
    Eigen::Vector3d _point;
};


int main(int argc, char** argv) {
    if (5 != argc)
    {
        cout << "usage: pose_estimation_3d3d img1 img2 depth1 depth2" << endl;
        return 1;
    }

    // read images
    Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    vector<KeyPoint> keypoints_1;
    vector<KeyPoint> keypoints_2;   // keypoints
    vector<DMatch> matches; // 匹配点
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "the number of matching feature points: " << matches.size() << endl;

    vector<Point3f> pts1, pts2;
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    Mat depth1 = imread(argv[3], CV_LOAD_IMAGE_UNCHANGED);  //深度图为16位无符号，单通道图像
    Mat depth2 = imread(argv[4], CV_LOAD_IMAGE_UNCHANGED);

    /*for (int i = 0; i < matches.size(); ++i)
    {
        ushort d1 = depth1.ptr<unsigned short>( int ( keypoints_1[matches[i].queryIdx].pt.y ) ) [int ( keypoints_1[matches[i].queryIdx].pt.x ) ];
        ushort d2 = depth2.ptr<unsigned short>( int ( keypoints_2[matches[i].trainIdx].pt.y ) ) [int ( keypoints_2[matches[i].trainIdx].pt.x ) ]
        Point2d p1 = pixel2cam(keypoints_1[matches[i].queryIdx].pt, K);
        Point2d p2 = pixel2cam(keypoints_2[matches[i].trainIdx].pt, K);
        pts1.push_back(Point3d(p1.x * d1, p1.y * d1, d1));
        pts2.push_back(Point3d(p2.x * d2, p2.y * d2, d2));
    }*/

    //  利用深度信息，对二维点建立其３Ｄ点对
    for (DMatch m:matches) {
        ushort d1 = depth1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
        ushort d2 = depth2.ptr<unsigned short>(int(keypoints_2[m.trainIdx].pt.y))[int(keypoints_2[m.trainIdx].pt.x)];
        if (0 == d1 || 0 == d2)
            continue; //bad depth
        Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        Point2d p2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
        float dd1 = float(d1) / 1000.0;     //深度尺度，单位由mm转换成m
        float dd2 = float(d2) / 1000.0;     //深度尺度，单位由mm转换成m
        pts1.push_back(Point3d(p1.x * dd1, p1.y * dd1, dd1));
        pts2.push_back(Point3d(p2.x * dd2, p2.y * dd2, dd2));
    }

    cout << "3d-3d pairs: " << pts1.size();
    Mat R, t;
    pose_estimation_3d3d(pts1, pts2, R, t);
    cout << "ICP via SVD results: " << endl;
    cout << "R = " << R << endl;    //第二张图到第一张图的转换旋转矩阵
    cout << "t = " << t << endl;    //第二张图到第一张图的平移向量
    cout << "R_inv = " << R.t() << endl;    // 第一张图到第二张图的转换旋转矩阵
    cout << "t_inv = " << -R.t() * t << endl;   //第一张图到第二张图的平移向量；T = [R,t;0,1], T逆=[R_inverse, -R_inverse*t; 0,1], R*R_transpose=E,R_inverse = R_transpose

    cout << "calling bundle adjustment " << endl;
    bundleAdjustment(pts1, pts2, R, t);

    for (int i = 0; i < 5; ++i)
    {
        cout << "p1 = " << pts1[i] << endl;
        cout << "p2 = " << pts2[i] << endl;
        cout << "(R*p2 + t) = " << R * (Mat_<double>(3,1) << pts2[i].x, pts2[i].y, pts2[i].z) + t << endl;
        cout << endl;
    }
}

// [u,v]--->[x,y,1]
Point2d pixel2cam(const Point2d& p, const Mat & K)
{
    return Point2d(
            (p.x - K.at<double>(0,2)) / K.at<double>(0,0),
            (p.y - K.at<double>(1,2)) / K.at<double>(1,1)
            );
}


// 线性代数求解，SVD singular value decomposition奇异值分解
/*
 * 【1】分别对两组点求取质心 p, p',计算两组点分区去质心后的坐标 qi = pi - p, qi' = pi' - p';
 * 【2】R = arg min{sum[(qi - R*qi')^2]}
 * [3] t = p - Rp'
 *
 * (qi - R * qi')^2 = qi转置 * qi   - 2*qi‘转置 * R * qi  + qi‘转置 * R转置* R * qi’ 第一项与R无关,
 * 因为R转置* R = I，第三项为qi‘转置 * qi’与R无关
 *
 *  所以，目标函数为负sum(qi转置 * R * qi);
 *  因为（qi转置 * R * qi）结果为一维度的实数，一个实数的迹（trace)等于自身，tr(realvalue) = realvalue;
 *  所以，qi转置 * R * qi = tr(R*qi转置*qi)；
 *  即 负sum(qi转置 * R * qi) = 负sum(tr(R*qi转置*qi)) ;
 *  根据迹的交换律性质，负sum(tr(R*qi转置*qi))= -tr(R*sum(qi转置*qi))
 *
 *  W= sum(qi * qi’转置) = U * 对角矩阵 * V转置   奇异值分解      R= U * V转置
 *   t =  p - R * p'
 * */
void pose_estimation_3d3d(const vector<Point3f>& pts1,
                          const vector<Point3f>& pts2,
                          Mat& R, Mat& t)
{
    Point3f p1, p2;     // center of mass
    int N = pts1.size();
    for (int i = 0; i < N ; ++i)
    {
        p1 += pts1[i];
        p2 += pts2[i];
    }

    p1 /= N;
    p2 /= N;

    // 去质心
    vector<Point3f> q1(N), q2(N);
    for (int j = 0; j < N; ++j)
    {
        q1[j] = pts1[j] - p1;
        q2[j] = pts2[j] - p2;
    }

    // 计算需要进行奇异值分解的 W = sum(qi * qi’转置) compute q1*q2^T

    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for (int k = 0; k < N; ++k)
    {
        W += Eigen::Vector3d(q1[k].x, q1[k].y, q1[k].z) * Eigen::Vector3d(q2[k].x, q2[k].y, q2[k].z).transpose();
    }

    cout << "W=" << W << endl;

    // SVD 分解
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU|Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    cout << "U=" << U << endl;
    cout << "V=" << V << endl;

    Eigen::Matrix3d R_ = U*(V.transpose());
    Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);

    // 由Eigen::vector转成 cv::Mat
    R = (Mat_<double>(3,3) <<
            R_(0,0), R_(0,1), R_(0,2),
            R_(1,0), R_(1,1), R_(1,2),
            R_(2,0), R_(2,1), R_(2,2)
    );

    t = (Mat_<double>(3,1) << t_(0,0), t_(1,0), t_(2,0));
}


void find_feature_matches(const Mat& img_1, const Mat& img_2,
                          std::vector<KeyPoint>& keypoints_1,
                          std::vector<KeyPoint>& keypoints_2,
                          std::vector<DMatch>& matches)
{

    // initialization
    Mat descriptors_1;
    Mat descriptors_2;
    Ptr<ORB> orb = ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);

    // detecting keypoints of oriented fast feature
    orb->detect(img_1, keypoints_1);
    orb->detect(img_2, keypoints_2);

    // computing brief descriptors according to keypoints
    orb->compute(img_1, keypoints_1, descriptors_1);
    orb->compute(img_2, keypoints_2, descriptors_2);

    vector<DMatch> initial_matches;
    BFMatcher matcher(NORM_HAMMING);
    matcher.match(descriptors_1, descriptors_2, initial_matches);

    double min_dist = 1000, max_dist = 0;
    for (int i = 0; i < descriptors_1.rows; ++i)
    {
        double dist = initial_matches[i].distance;
        if (dist > max_dist)
            max_dist = dist;
        if (dist < min_dist)
            min_dist = dist;
    }

    printf("--Max dist: %f \n", max_dist);
    printf("--Min dist: %f \n", min_dist);

    for (int j = 0; j < initial_matches.size(); ++j)
    {
        if (initial_matches[j].distance < max(2 * min_dist, 30.0))
        {
            matches.push_back(initial_matches[j]);
        }
    }
}


void bundleAdjustment(const vector<Point3f>& pts1,
                      const vector<Point3f>& pts2,
                      Mat& R, Mat& t)
{
    // g2o initilization
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,3> > Block; // PoseDim = 6, LandmarkDim =3
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverEigen<Block::PoseMatrixType>();    // 线性方程求解器
    Block* solver_ptr = new Block(linearSolver);    //矩阵块求解器 矩阵分解
    g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);  //迭代优化算法
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // vertex
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap(); // 优化变量，相机位姿
    pose->setId(0);
    pose->setEstimate(g2o::SE3Quat(
            Eigen::Matrix3d::Identity(), Eigen::Vector3d(0,0,0)
    )); //顶点赋初始值

    optimizer.addVertex(pose);  //添加顶点


    //edges
    int index = 1;
    vector<EdgeProjectXYZRGBDPoseOnly*> edges;
    for (size_t i = 0; i < pts1.size(); ++i)
    {
        EdgeProjectXYZRGBDPoseOnly* edge = new EdgeProjectXYZRGBDPoseOnly(Eigen::Vector3d(pts2[i].x, pts2[i].y, pts2[i].z)); //构建边，参数为点2带进去赋值给估计值
        edge->setId(index);
        edge->setVertex(0, dynamic_cast<g2o::VertexSE3Expmap*> (pose)); //设置边链接的顶点
        edge->setMeasurement(Eigen::Vector3d(pts1[i].x, pts1[i].y, pts1[i].z)); //点1 = T * 点2 = 点1'
        edge->setInformation(Eigen::Matrix3d::Identity() * 1e4);    //误差项系数矩阵  信息矩阵
        optimizer.addEdge(edge);    //向图中添加边
        index++;
        edges.push_back(edge);  //保存所有的边
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 -t1);
    cout << "optimization costs time: " << time_used.count() << "seconds." << endl;

    cout << endl << "after optimzation: " <<endl;
    cout << "T=" << endl << Eigen::Isometry3d(pose->estimate()).matrix() << endl;
}
















