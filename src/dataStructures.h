#ifndef dataStructures_h
#define dataStructures_h

#include <vector>
#include <opencv2/core.hpp>


struct DataFrame { // represents the available sensor information at the same time instance
    
    cv::Mat cameraImg; // camera image
    
    std::vector<cv::KeyPoint> keypoints; // 2D keypoints within camera image
    cv::Mat descriptors; // keypoint descriptors
    std::vector<cv::DMatch> kptMatches; // keypoint matches between previous and current frame
};

// direct drop-in replacement for assignment 1 (supporting api/usage shortfalls and all)
template<typename T>
class DataBuffer {
public:
    using iterator = typename std::vector<T>::iterator;

    using value_type = T;
    using differenct_type = std::ptrdiff_t;
    using reference = value_type&;
    using pointer = T*;
    using const_reference = const reference;
    using iterator_category = std::random_access_iterator_tag;

    DataBuffer(std::size_t _cap) : 
        capacity(_cap), 
        used(0) {
        data.reserve(_cap);
        value_type t;
        data.assign(_cap, t);
    };

    std::size_t size() { return used; }
    iterator end() { return data.end(); }
    iterator begin() { return data.end() - used; }

    void push_back(const_reference itm) {
        // slide everything down
        std::size_t offset = capacity - used;
        for(int i = 0; i < used; i++) {
            if (i + offset > 0) {
                data[i + offset - 1] = std::move(data[i + offset]);
            }
        }
        data[capacity - 1] = std::move(itm);
        used = std::min(capacity, used + 1);
    }

private:
    std::vector<value_type> data;
    std::size_t capacity;
    std::size_t used;
};


#endif /* dataStructures_h */
