#define CPU_ONLY // comment this line if you want to use GPU
#include "../include/Classifier.h"
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace caffe;
using namespace cv;
using namespace std;

Classifier::Classifier(const string& model_file, const string& trained_file) {
#ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
#else
    Caffe::set_mode(Caffe::GPU);
#endif

    /* Load the network. */
    net_.reset(new Net<float>(model_file, TEST));
    net_->CopyTrainedLayersFrom(trained_file);

    //CHECK_EQ(net_->num_inputs(),1)<<"Network should have exactly one input.";
    //CHECK_EQ(net_->num_outputs(),1)<<"Network should have exactly one output.";

    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    //CHECK(num_channels_ == 3 || num_channels_ == 1)
    //    << "Input layer should have 1 or 3 channels.";
    input_geometry_ = Size(input_layer->width(), input_layer->height());

    /* Load the binaryproto mean file. */
    //SetMean(mean_file);
}

vector<float> Classifier::Predict(const Mat& img) {
    Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_, input_geometry_.height,input_geometry_.width);
    net_->Reshape();

    vector<Mat> input_channels;
    WrapInputLayer(&input_channels);

    Preprocess(img, &input_channels); //convert img to caffe input

    Timer tmr;
    tmr.Start();
    net_->Forward(); // Caffe functions
    tmr.Stop();
    cout<<"Feature extracton time ="<<tmr.Seconds()<<endl;

    /* Copy the output layer to std::vector */
    caffe::shared_ptr< Blob<float> > output_layer = net_-> blob_by_name("fc8");

    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->channels();

    //debug
    vector<string> tmpnames = net_-> blob_names();
    for (int i=0; i<tmpnames.size(); i++)
        cout<<tmpnames[i]<<" "<<endl;

    vector<float> temp = vector<float>(begin, end);
    /*cout<<endl<<"Feature:"<<endl;
    for (vector<float>::const_iterator i = temp.begin(); i != temp.end(); ++i) {
        cout << *i << ' ';
    }*/
    cout<<endl<<"FeatDim="<<temp.size()<<"  "<<"Feature done!"<<endl;
    return temp;
}

void Classifier::Preprocess(const Mat& img, vector<Mat>* input_channels) {
    Mat sample;
    if (img.channels() == 3 && num_channels_ == 1)
        cvtColor(img, sample, COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cvtColor(img, sample, COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cvtColor(img, sample, COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cvtColor(img, sample, COLOR_GRAY2BGR);
    else
        sample = img;

    Mat sample_resized;
    if (sample.size() != input_geometry_)
        resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;

    Mat sample_float;
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);

    //Mat sample_normalized;
    //subtract(sample_float, mean_, sample_normalized);
    split(sample_float, *input_channels);

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data) == net_->input_blobs()[0]->cpu_data())
        << "Input channels are not wrapping the input layer of the network. ";

}

void Classifier::WrapInputLayer(vector<Mat>* input_channels) {
    Blob<float>* input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height= input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int i=0; i<input_layer->channels();++i) {
        Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

extern "C" void* ClassifierInit(const char *model_file, const char *trained_file)
{
    //string model_file = "./vgg_face_caffe/VGG_FACE_deploy.prototxt";
    //string trained_file = "./vgg_face_caffe/VGG_FACE.caffemodel";
    Classifier *pClassifier = new Classifier(model_file, trained_file);    
    return pClassifier;
}

extern "C" void ClassifierRelease(void *pClassifier)
{
    delete (Classifier*)pClassifier;
}

template <typename T>
cv::Mat createMat(T* data, int rows, int cols, int chs = 3) {
    cv::Mat mat(rows, cols, CV_MAKETYPE(cv::DataType<T>::type-1, chs));
    memcpy(mat.data, data, (size_t)rows * cols * chs * sizeof(T));
    return mat;
}

extern "C" float ClassifierMatching(void *pClassifier, int rowsa, int colsa, char* imga, int rowsb, int colsb, char* imgb)
{
    Mat matac3, matbc3;
    matac3 = createMat(imga, rowsa, colsa);
    matbc3 = createMat(imgb, rowsb, colsb);

    Mat avgA(matac3.rows, matac3.cols, CV_8UC3, Scalar(93.5940,104.7624,129.1863));
    Mat imageA = matac3 - avgA;
    Mat avgB(matbc3.rows, matbc3.cols, CV_8UC3, Scalar(93.5940,104.7624,129.1863));
    Mat imageB = matbc3 - avgB;

    vector<float> outputA = ((Classifier*)pClassifier)->Predict(imageA);
    vector<float> outputB = ((Classifier*)pClassifier)->Predict(imageB);

    float in_prod = 0;
    for (int i=0; i<outputA.size(); i++)
        in_prod += outputA[i]*outputB[i];
    float sim = in_prod/(norm(outputA,NORM_L2)*norm(outputB,NORM_L2));   

    return sim;     
}
