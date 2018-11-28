//
// Created by zbox on 18-10-31.
//

#include <iostream>
using namespace std;

//opencv
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

// DboW3
#include "DBoW3/DBoW3.h"

#include <string>
#include <vector> // ?


int main(int argc, char** argv)
{
    // 读取图像，提取orb特征，利用DBoW3制作字典
    cout << "reading images..." << endl;
    vector<Mat> images;

    for (int i = 0; i < 10; ++i)
    {
        images.push_back( cv::imread("./data/" + to_string(i+1) + ".png") );
    }

    cout << "detecting ORB features ... " << endl;
    Ptr<Feature2D> detector = ORB::create();
    vector<Mat> descriptors;
    for (Mat& image:images)
    {
        vector<KeyPoint> keypoints; // 一张图片会创建多个keypoint
        Mat descriptor;
        detector->detectAndCompute(image, Mat(), keypoints, descriptor);
        descriptors.push_back(descriptor);
    }

    cout <<  "creating vocabulary ..." << endl;
    DBoW3::Vocabulary vocab;
    vocab.create(descriptors);
    cout<<"vocabulary info: "<<vocab<<endl;
    vocab.save("vocabulary.yml.gz");
    cout <<"done"<<endl;
    return 0;
}

