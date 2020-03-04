#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
eval_stats matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
    std::vector<cv::DMatch> &matches, int normType, std::string matcherType, std::string selectorType)
{
    eval_stats stats;
    // configure matcher
    // bool crossCheck = false;
    bool crossCheck = true;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        // int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
        // int normType = cv::NORM_L2;
        // if (descriptorType.compare("ORB") == 0 || descriptorType.compare("BRISK") == 0 ||
        //     descriptorType.compare("BRIEF") == 0) {
        //     // normType = cv::NORM_HAMMING2;
        //     normType = cv::NORM_HAMMING;
        // }
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        // OpenCV bug workaround : convert binary descriptors to floating point due to bug in current OpenCV implementation
        if (descSource.type() != CV_32F) {
            descSource.convertTo(descSource, CV_32F);
        }
        if (descSource.type() != CV_32F) {
            descRef.convertTo(descRef, CV_32F);
        }
        // for ORB
        // cv::FlannBasedMatcher matcher = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
        cv::Ptr<cv::flann::IndexParams> indexParams;
        if (normType == cv::NORM_L2) {
            indexParams = new cv::flann::KDTreeIndexParams();
        } else if (normType == cv::NORM_HAMMING) {
            indexParams = new cv::flann::LshIndexParams(20, 15, 2);
        } else {
            cerr << "unsupported normType" << endl;
            exit(1);
        }

        // matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        matcher = new cv::FlannBasedMatcher(indexParams);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)
        double t = (double)cv::getTickCount();
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
        stats.time = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
        vector<vector<cv::DMatch>> knn_matches;
        double t = (double)cv::getTickCount();
        matcher->knnMatch(descSource, descRef, knn_matches, 2);  // find at least 2 best matches
        stats.time = ((double)cv::getTickCount() - t) / cv::getTickFrequency();   

        double minDescDistRatio = 0.8;
        for (auto &m : knn_matches) {
            if (m.size() >= 2 && m[0].distance < minDescDistRatio*m[1].distance) {
                matches.push_back(m[0]);
            }
        }   
    }

    stats.points = matches.size();
    return stats;
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
// required:  BRIEF, ORB, FREAK, AKAZE and SIFT
eval_stats descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType, int &normType)
{
    eval_stats stats;
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;

    if (descriptorType.compare("BRISK") == 0) {
        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.
        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    } else if (descriptorType.compare("BRIEF") == 0) {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    } else if (descriptorType.compare("ORB") == 0) {
        extractor = cv::ORB::create(2000);
    } else if (descriptorType.compare("FREAK") == 0) {
        extractor = cv::xfeatures2d::FREAK::create();
    } else if (descriptorType.compare("AKAZE") == 0) {
        extractor = cv::AKAZE::create();
    } else if (descriptorType.compare("SIFT") == 0) {
        extractor = cv::xfeatures2d::SIFT::create();
    } else {
        cerr << "descKeypoints -  unexpected detector type: " << descriptorType << endl;
        exit(-1);
    }

    // let the extractor determine the norm
    normType = extractor->defaultNorm();

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
    
    stats.time = t;
    stats.points = 0;

    return stats;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
eval_stats implCornerDetection(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis, bool useHarris)
{
    eval_stats stats;
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints
    int harrisCorner = 4;

    string msg("Shi-Tomasi");
    if (useHarris) {
        msg = "Harris";
        // https://stackoverflow.com/questions/54720646/what-does-ksize-and-k-mean-in-cornerharris
        // TLDR;  bigger k, less false corners;  smaller k, lot more corners but more false positives.
        harrisCorner = 4;
    }
    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, useHarris, harrisCorner);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {
        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << msg << " detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = msg + " Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
    
    stats.time = t;
    stats.points = keypoints.size();

    return stats;
}

eval_stats detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis) {
    return implCornerDetection(keypoints, img, bVis, false);
}

eval_stats detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis) {
    return implCornerDetection(keypoints, img, bVis, true);
}

// FAST, BRISK, ORB, FREAK, AKAZE, SIFT
eval_stats detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis){
    eval_stats stats;    
    double t;
    cv::Ptr<cv::FeatureDetector> detector;

    if (detectorType.compare("FAST") == 0) {
        detector = cv::FastFeatureDetector::create(10, true);
    } else if (detectorType.compare("BRISK") == 0) {
        detector = cv::BRISK::create();
    } else if (detectorType.compare("ORB") == 0) {
        // combination of FAST and BRIEF
        // can set the max number of keypoints
        detector = cv::ORB::create(2000);
    } else if (detectorType.compare("AKAZE") == 0) {
        detector = cv::AKAZE::create();
        t = (double)cv::getTickCount();
    } else if (detectorType.compare("SIFT") == 0) {
        detector = cv::xfeatures2d::SIFT::create();
    } else {
        cerr << "detKeypoints - unexpected detector type: " << detectorType << endl;
        exit(1);
    }

    t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    stats.time = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = detectorType + " Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
    
    stats.points = keypoints.size();

    return stats;
}