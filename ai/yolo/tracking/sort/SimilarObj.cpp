#include "SimilarObj.h"

#include "opencv2/objdetect.hpp"        //hog

#ifdef CONFIG_SPDLOG
#define LOG_TAG "SimObj"
#include "log.h"
#endif

#define SSIM_FEATURE_SIZE   (8)

#define CLAMP(value, mmin, mmax) (mmin < mmax           \
  ? (value < mmin ? mmin : value > mmax ? mmax : value) \
  : (value < mmax ? mmax : value > mmin ? mmin : value) \
)

static bool check_box_valid(const cv::Rect_<float> &bb)
{
    if (bb.x >= 0 && bb.y >= 0) {
        return true;
    } else {
        //bb.x == nan
        return false;
    }
}

// Computes IOU between two bounding boxes
static float GetIOU(cv::Rect_<float> bb_test, cv::Rect_<float> bb_gt)
{
    float in = (bb_test & bb_gt).area();
    float un = bb_test.area() + bb_gt.area() - in;
    if (un < DBL_EPSILON) {
        return 0;
    }
    return (float)(in / un);
}

SimilarObj::SimilarObj(void)
{
    inited_ = 0;
}

SimilarObj::~SimilarObj(void)
{

}

static cv::Scalar calc_image_ssim( const cv::Mat& i1, const cv::Mat& i2)
{
    const double C1 = 6.5025, C2 = 58.5225;
    /***************************** INITS **********************************/
    int d = CV_32F;

    cv::Mat I1, I2;
    i1.convertTo(I1, d);            // cannot calculate on one byte large values
    i2.convertTo(I2, d);

    cv::Mat I2_2   = I2.mul(I2);        // I2^2
    cv::Mat I1_2   = I1.mul(I1);        // I1^2
    cv::Mat I1_I2  = I1.mul(I2);        // I1 * I2

    /*************************** END INITS **********************************/
    cv::Mat mu1, mu2;                   // PRELIMINARY COMPUTING
    cv::GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);

    cv::Mat mu1_2   =   mu1.mul(mu1);
    cv::Mat mu2_2   =   mu2.mul(mu2);
    cv::Mat mu1_mu2 =   mu1.mul(mu2);

    cv::Mat sigma1_2, sigma2_2, sigma12;

    cv::GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;

    cv::GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;

    cv::GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    ///////////////////////////////// FORMULA ////////////////////////////////
    cv::Mat t1, t2, t3;

    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);                 // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);                 // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

    cv::Mat ssim_map;
    cv::divide(t3, t1, ssim_map);        // ssim_map =  t3./t1;

    cv::Scalar mssim = mean(ssim_map);   // mssim = average of ssim map
    return mssim;
}

void SimilarObj::generateSsimImage(cv::Mat &in, cv::Rect_<float> box, cv::Mat &out)
{
    int x = box.x;
    int y = box.y;
    int w = box.width;
    int h = box.height;
    w = CLAMP(w, SORT_BOX_MIN, SORT_YOLO_SIZE-2);
    h = CLAMP(h, SORT_BOX_MIN, SORT_YOLO_SIZE-2);

    if (x+w >= SORT_YOLO_SIZE) {
        x = SORT_YOLO_SIZE-1-x;
    }
    if (y+h >= SORT_YOLO_SIZE) {
        y = SORT_YOLO_SIZE-1-h;
    }

    // crop,rescale, sub-mean
    cv::Mat crop = cv::Mat(in, cv::Rect(x, y, w, h));
    cv::resize(crop, out, cv::Size(SSIM_FEATURE_SIZE, SSIM_FEATURE_SIZE), 0, 0, cv::INTER_LINEAR);
}

void SimilarObj::generateHogImage(cv::Mat &in, cv::Rect_<float> box, cv::Mat &out)
{
    int x = box.x;
    int y = box.y;
    int w = box.width ;
    int h = box.height;
    w = std::max(20, w);
    h = std::max(20, h);
    if (x+w >= SORT_YOLO_SIZE) {
        x = SORT_YOLO_SIZE-1-x;
    }
    if (y+h >= SORT_YOLO_SIZE) {
        y = SORT_YOLO_SIZE-1-h;
    }
    LOGD("x,y,w,h:%d %d %d %d\n", x,y,w,h); //50~80

    cv::Mat resize;
    cv::Mat crop = cv::Mat(in, cv::Rect(x, y, w, h));
    cv::resize(crop, resize, cv::Size(40, 40), 0, 0, cv::INTER_LINEAR);

    //cv::HOGDescriptor hog;    //aqual to HOGDescriptor(Size(64,128), Size(16,16), Size(8,8), Size(8,8), 9, 1 )

    //HOGDescriptor (Size _winSize, Size _blockSize, Size _blockStride, Size _cellSize, int _nbins, 
    //  int _derivAperture=1, double _winSigma=-1, int _histogramNormType=HOGDescriptor::L2Hys, 
    //double _L2HysThreshold=0.2, bool _gammaCorrection=false,
    //int _nlevels=HOGDescriptor::DEFAULT_NLEVELS, bool _signedGradient=false)
    cv::HOGDescriptor hog = cv::HOGDescriptor(cv::Size(20, 20), cv::Size(10, 10), cv::Size(5, 5), cv::Size(5, 5), 9);

    // hog.blockSize = 16;
    // hog.cellSize = 4;
    // hog.blockStride = 8;
    std::vector<float> ders;
    std::vector<cv::Point> locs;
    cv::Mat mat_gray;
    cvtColor(resize, mat_gray, cv::COLOR_BGR2GRAY);
    hog.compute(mat_gray, ders, cv::Size(10, 10), cv::Size(0, 0), locs);    //Window stride. It must be a multiple of block stride
    cv::Mat Hogfeat(ders.size(), 1, CV_32FC1);
    for (int i=0; i < (int) ders.size(); i++) {
        Hogfeat.at<float>(i,0)=ders.at(i);
    }
    out = Hogfeat;
}

void SimilarObj::init(cv::Mat &frame, const TrackingBox &tbox, int id)
{
    sort_id_ = tbox.id;
    tbox_    = tbox;
    tbox_.id = id;

#if USE_SSIM
    generateSsimImage(frame, tbox.box, mat_roi_);
#else
    generateHogImage(frame, tbox.box, mat_roi_);
#endif

    inited_ = 1;
}

void SimilarObj::updateBox(const TrackingBox &tbox)
{
    assert(tbox_.class_idx == tbox.class_idx);

    tbox_.box = tbox.box;
    tbox_.frame = tbox.frame;
    tbox_.confidence = tbox.confidence;
}

void SimilarObj::update(cv::Mat &frame, const TrackingBox &tbox)
{
    updateBox(tbox);

#if USE_SSIM
    generateSsimImage(frame, tbox.box, mat_roi_);
#else
    generateHogImage(frame, tbox.box, mat_roi_);
#endif
}

float SimilarObj::checkMatchScore(cv::Mat &frame, const RectBox &box)
{
    if (!inited_) {
        return -1;
    }
#if USE_SSIM
    if (mat_roi_.cols >= SSIM_FEATURE_SIZE) {
        cv::Mat in2;
        generateSsimImage(frame, box, in2);
        return calc_image_ssim(mat_roi_, in2)[0];
    }
#else
    if (mat_roi_.rows > 0) {
        cv::Mat in2;
        generateHogImage(frame, box, in2);
        // This is for comparing the HOG features of two images without using any SVM 
        // (It is not an efficient way but useful when you want to compare only few or two images)
        // Simple distance
        // Consider you have two HOG feature vectors for two images Hogfeat1 and Hogfeat2 and those are same size.
        assert(in2.rows == mat_roi_.rows);
        assert(in2.cols == mat_roi_.cols);
        float distance = 0;
        for(int i = 0; i < mat_roi_.rows; i++) {
            distance += abs(mat_roi_.at<float>(i, 0) - in2.at<float>(i, 0));
        }
        distance /= mat_roi_.rows;
        return distance;
    }
#endif
    return -1; //incase
}