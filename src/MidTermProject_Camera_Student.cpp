/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

    // -d <DET_TYPE> -m <MAT_TYPE> -s <SEL_TYP> [-v[isible]] [-f[ocusOnVehicle]] [-l[imitKpts]]
    // DET_TYPE:  SHITOMASI, HARRIS, FAST, BRISK, ORB, FREAK, AKAZE, SIFT
    // MAT_TYPE:  MAT_BF, MAT_FLANN
    // DES_TYPE:  BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
    // SEL_TYPE:  SEL_NN, SEL_KNN

void usage(const char *progname) {
    cout << "usage: " << endl;
    cout << progname << " -d <DETECTOR_TYPE> -m <MATCHER_TYPE> -x <DESCRIPTOR_TYPE> -s <SELECTOR_TYPE> \\" << endl;
    cout << "    [-v] [-f] [-l]" << endl;
    cout << "-v: visualize results" << endl;
    cout << "-f: focusOnVehicle" << endl;
    cout << "-l: limitKpts" << endl;    
    cout << "" << endl;
    cout << "DETECTOR_TYPE:  SHITOMASI, HARRIS, FAST, BRISK, ORB, FREAK, AKAZE, SIFT" << endl;
    cout << "MATCHER_TYPE:  MAT_BF, MAT_FLANN" << endl;
    cout << "DESCRIPTOR_TYPE: BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT" << endl;
    cout << "SELECTOR_TYPE:  SEL_NN, SEL_KNN" << endl;
    cout << "";
    cout << "Example:" << endl;
    cout << "  ./2D_feature_tracking -d SHITOMASI -m MAT_BF -x BRISK -s SEL_NN" << endl;
}

/* MAIN PROGRAM */
eval_summary _main(int argc, const char *argv[])
{
    /* INIT VARIABLES AND DATA STRUCTURES */
    string detectorType = "";    // SHITOMASI, HARRIS, FAST, BRISK, ORB, FREAK, AKAZE, SIFT
    string matcherType = "";     // MAT_BF, MAT_FLANN
    string descriptorType = "";  // BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
    string selectorType = "";    // SEL_NN, SEL_KNN

    bool bVis = false;            // visualize results
    bool bFocusOnVehicle = false;
    bool bLimitKpts = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-d") == 0) {
            detectorType = argv[++i];
            cout << "DetectorType: " << detectorType << endl;
        } else if (strcmp(argv[i], "-m") == 0) {
            matcherType = argv[++i];
            cout << "MatcherType: " << matcherType << endl;
        } else if (strcmp(argv[i], "-x") == 0) {
            descriptorType = argv[++i];
            cout << "DescriptorType: " << descriptorType << endl;
        } else if (strcmp(argv[i], "-s") == 0) {
            selectorType = argv[++i];
            cout << "SelectorType: " << selectorType << endl;
        } else if (strncmp(argv[i], "-v", 2) == 0) {
            bVis = true;
            printf("\bvisualize: %d", bVis);
        } else if (strncmp(argv[i], "-f", 2) == 0) {
            bFocusOnVehicle = true;
            printf("\bfocusOnVehicle: %d", bFocusOnVehicle);   
        } else if (strncmp(argv[i], "-l", 2) == 0) {
            bLimitKpts = true;
            printf("\blimitKpts: %d", bLimitKpts);         
        } else {
            cout << "unexpected argument found: " << argv[i] << endl;
            exit(-1);
        }
    }

    if (detectorType == "" || matcherType == "" || descriptorType == "" || selectorType == "") {
        cout << "incomplete arguments given." << endl;
        usage(argv[0]);
        exit(-1);
    }

    eval_stats stats;
    eval_summary summary;

    summary.detector_type = detectorType;
    summary.matcher_type = matcherType;
    summary.descriptor_type = descriptorType;
    summary.selector_type = selectorType;

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    // vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    DataBuffer<DataFrame> dataBuffer(2);

    /* MAIN LOOP OVER ALL IMAGES */
    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        //// STUDENT ASSIGNMENT
        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;
        dataBuffer.push_back(frame);

        //// EOF STUDENT ASSIGNMENT
        cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image

        //// STUDENT ASSIGNMENT
        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
        //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
        try {
            if (detectorType.compare("SHITOMASI") == 0) {
                stats = detKeypointsShiTomasi(keypoints, imgGray, bVis);
            } else if (detectorType.compare("HARRIS") == 0) {
                stats = detKeypointsHarris(keypoints, imgGray, bVis);
            } else {
                stats = detKeypointsModern(keypoints, imgGray, detectorType, bVis);
            }
            summary.detect_time[imgIndex] = stats.time;
            summary.detect_points[imgIndex] = stats.points;
        } catch (exception &e) {
            cerr << "Exception occurred while processing keypoints: " << e.what() << endl;
        }

        //// EOF STUDENT ASSIGNMENT

        //// STUDENT ASSIGNMENT
        //// TASK MP.3 -> only keep keypoints on the preceding vehicle

        // only keep keypoints on the preceding vehicle
        cv::Rect vehicleRect(535, 180, 180, 150);
        if (bFocusOnVehicle)
        {
            // ...
        }

        //// EOF STUDENT ASSIGNMENT

        // optional : limit number of keypoints (helpful for debugging and learning)
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;
        cout << "#2 : DETECT KEYPOINTS done" << endl;

        /* EXTRACT KEYPOINT DESCRIPTORS */

        //// STUDENT ASSIGNMENT
        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
        //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

        cv::Mat descriptors;
        try {
            stats = descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);
            summary.description_time[imgIndex] = stats.time;
        } catch (exception &e) {
            cerr << "Exception occurred while processing descriptors: " << e.what() << endl;
        }
        
        //// EOF STUDENT ASSIGNMENT

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {
            /* MATCH KEYPOINT DESCRIPTORS */

            //// STUDENT ASSIGNMENT
            //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
            //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

            vector<cv::DMatch> matches;
            try {
                stats = matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                                (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                                matches, descriptorType, matcherType, selectorType);
                summary.match_time[imgIndex] = stats.time;
                summary.match_points[imgIndex] = stats.points;
            } catch (exception &e) {
                cerr << "Exception occurred while processing matches: " << e.what() << endl;
            }

            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

            // visualize matches between current and previous image
            if (bVis)
            {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;
                cv::waitKey(0); // wait for key to be pressed
            }
        }

    } // eof loop over all images

    return summary;
}

int batch_main(int argc, const char *argv[]) {
    vector<string> detectors =  { "SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "FREAK", "AKAZE", "SIFT" };
    vector<string> matchers =  { "MAT_BF", "MAT_FLANN" };
    vector<string> descriptors =  { "BRISK", "BRIEF", "ORB", "FREAK", "AKAZE", "SIFT" };
    vector<string> selectors =  { "SEL_NN", "SEL_KNN" };

    const char *args[9];
    args[0] = argv[0];
    args[1] = "-d";
    args[3] = "-x";
    args[5] = "-m";
    args[7] = "-s";

    vector<eval_summary> summaries;

    for (auto det : detectors) {
        for (auto des : descriptors) {
            for (auto mat: matchers) {
                for (auto sel: selectors) {
                    args[2] = det.c_str();
                    args[4] = des.c_str();
                    args[6] = mat.c_str();
                    args[8] = sel.c_str();

                    summaries.push_back(_main(9, args));
                }
            }
        }
    }

}

int main(int argc, const char *argv[]) {
    bool is_batch = false;
    // scan for -b
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "-b") == 0) {
            is_batch = true;
            break;
        }
    }

    if (is_batch) {
        batch_main(argc, argv);
    } else {
        eval_summary summary = _main(argc, argv);  
        cout << "Summary:" << endl;
        cout << " Detector Type: " << summary.detector_type << endl;
        cout << " Matcher Type: " << summary.matcher_type << endl;
        cout << " Descriptor Type: " << summary.descriptor_type << endl;
        cout << " Selector Type: " << summary.selector_type << endl;

        for (int i = 1; i < MAX_EVALS; i++) {
            cout << "detect_time: " << summary.detect_time[i] << " points: " << summary.detect_points[i] << endl;
            cout << "match_time: " << summary.match_time[i] << " points: " << summary.match_points[i] << endl;
        }
    }

    return 0;
}