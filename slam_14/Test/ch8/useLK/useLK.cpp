//
// Created by zbox on 18-8-15.
//

#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <chrono>
using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>


int main(int argc, char** argv)
{
    if (2 != argc)
    {
        cout << "usage: useLK path_to_dataset"<< endl;
        return 1;
    }

    string path_to_dataset = argv[1];
    string associate_file = path_to_dataset + "/associate.txt";

    ifstream fin(associate_file);
    if( !fin )
    {
        cerr << "I cannot find associate.txt" << endl;
        return 1;
    }

    string rgb_file, depth_file, time_rgb, time_depth;
    list<cv::Point2f> keypoints;
    cv::Mat color, depth, last_color;

    for (int index = 0; index < 100; ++index)
    {
        fin >> time_rgb >> rgb_file >> time_depth >> depth_file;
        color = cv::imread(path_to_dataset + "/" + rgb_file);
        depth = cv::imread(path_to_dataset + "/" + depth_file, -1);
        if (0 == index)
        {
            vector<cv::KeyPoint> kps;
            cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
            detector->detect(color, kps);
            for (auto kp:kps)
            {
                keypoints.push_back(kp.pt); // keypoints起初保存了第一张图片中的所有关键点；随着光流法跟踪后续图片，将跟踪到的关键点更新的旧的keypoints，同时根据status失败位置对应删除掉keypoints的关键点，参见line78
            }
            last_color = color;
            continue;
        }

        if (nullptr == color.data || nullptr == depth.data)
        {
            continue;
        }

        vector<cv::Point2f> next_keypoints;
        vector<cv::Point2f> pre_keypoints;
        for (auto kp:keypoints)
        {
            pre_keypoints.push_back(kp);
        }
        vector<unsigned char> status;
        vector<float> error;
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        cv::calcOpticalFlowPyrLK(last_color, color, pre_keypoints, next_keypoints, status, error);
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 -t1);
        cout << "LK Flow use time: " << time_used.count()<< "seconds." << endl;

        // 光流法跟踪上一时刻的keypoints在当前时刻图片中的位置，将跟踪到的关键点更新到旧的keypoints，同时根据status失败位置对应删除掉keypoints的对应位置关键点
        int i = 0;
        for (auto iter = keypoints.begin(); iter != keypoints.end(); i++)
        {
            if (status[i] == 0)
            {
                iter = keypoints.erase(iter); // 光流法跟踪上一时刻的keypoints在当前时刻图片中的位置，根据status失败位置对应删除掉keypoints的对应位置关键点
                continue;
            }

            *iter = next_keypoints[i];  // 光流法跟踪上一时刻的keypoints在当前时刻图片中的位置，将跟踪到的关键点更新到旧的keypoints
            iter++;
        }

        cout << "tracked keypoints: " << keypoints.size() << endl;
        if (keypoints.size() == 0)
        {
            cout << "all keypoints are lost." << endl;
            break;
        }

        cv::Mat img_show = color.clone();
        for (auto kp:keypoints)
        {
            cv::circle(img_show, kp, 10,cv::Scalar(0,240,0), 1);    //在img_show上绘制kp,大小为10， 颜色为Scalar
        }
        cv::imshow("corners", img_show);
        cv::waitKey(0);
        last_color = color;     // 将当前图片替换成上一张图片，即将读取下一时刻的图片到color中
    }

    return 0;
}



























