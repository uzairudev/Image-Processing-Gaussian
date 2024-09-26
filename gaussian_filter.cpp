// 
// g++ gaussian_filter.cpp -fopenmp `pkg-config opencv4 --cflags` -c
// g++ gaussian_filter.o  -fopenmp `pkg-config opencv4 --libs` -lstdc++ -o gaussian_filter

// # Use this to run 
// # ./gaussian_filter 
//  choose Kernel size and Sigma. 


#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

#define M_PI  3.14159265358979323846

double **get_Gaussian_Filter(int kernel_size, double sigma){
  double **result = new double*[kernel_size];
  for(int i = 0; i < kernel_size; i++)
    result[i] = new double[kernel_size];
 

  // Calculate Gaussian kernel coefficients
  double sum = 0.0;
  int center = (kernel_size - 1) / 2;
  for (int i = 0; i < kernel_size; i++) {
    for (int j = 0; j < kernel_size; j++) {
        double val = ((i - center) * (i - center)) + ((j - center) * (j - center));
      double exponent =  (-1.0 * val) / (2.0 * sigma * sigma);
      result[i][j] = exp(exponent) / (2.0 * M_PI * sigma * sigma);
      sum += result[i][j];
    }
  }

  cout << sum << endl;

  // Normalize kernel coefficients
  for (int i = 0; i < kernel_size; i++) {
    for (int j = 0; j < kernel_size; j++) {
      result[i][j] /= sum;
    }
  }
  return result;
}

cv::Mat convolveGaussianFilter(double** filter, int filterSize, cv::Mat rgb_image) {
    cv::Mat convolved_image(rgb_image.size(), rgb_image.type());

    // Convert the RGB image to a grayscale image
    cv::Mat gray_image;
    cv::cvtColor(rgb_image, gray_image, cv::COLOR_BGR2GRAY);

    // Iterate over the pixels of the gray image
    auto begin = chrono::high_resolution_clock::now();
    const int iter = 100;
    for (int it=0;it<iter;it++){

        #pragma omp parallel for
            for (int i = 0; i < gray_image.rows; i++) {
                for (int j = 0; j < gray_image.cols; j++) {
                    double sum_r = 0;
                    double sum_g = 0;
                    double sum_b = 0;
                    double sum_filter = 0;

                    // Iterate over the pixels of the filter
                    for (int k = 0; k < filterSize; k++) {
                        for (int l = 0; l < filterSize; l++) {
                            // Compute the coordinates of the current pixel in the image
                            int x = i + k - filterSize/2;
                            int y = j + l - filterSize/2;

                            // Make sure the pixel is within the image bounds
                            if (x >= 0 && x < gray_image.rows && y >= 0 && y < gray_image.cols) {
                                // Compute the convolved value for each channel
                                double filter_value = filter[k][l];
                                sum_filter += filter_value;

                                sum_r += filter_value * rgb_image.at<cv::Vec3b>(x, y)[2];
                                sum_g += filter_value * rgb_image.at<cv::Vec3b>(x, y)[1];
                                sum_b += filter_value * rgb_image.at<cv::Vec3b>(x, y)[0];
                            }
                        }
                    }

                    // Normalize the convolved values and set them in the convolved image
                    convolved_image.at<cv::Vec3b>(i, j)[2] = cv::saturate_cast<uchar>(sum_r/sum_filter);
                    convolved_image.at<cv::Vec3b>(i, j)[1] = cv::saturate_cast<uchar>(sum_g/sum_filter);
                    convolved_image.at<cv::Vec3b>(i, j)[0] = cv::saturate_cast<uchar>(sum_b/sum_filter);
                }
            }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - begin;

    cout << "Total time: " << diff.count() << " s" << endl;
    cout << "Time for 1 iteration: " << diff.count() / iter << " s" << endl;
    cout << "IPS: " << iter / diff.count() << endl;

    return convolved_image;
}


int main()
{
    // Load the stereo image
    Mat stereo_image = imread("stereo_image.jpg");

    // Get parameters from user input
    int kernel_size, sigma;
    cout << "Enter kernel size (must be an odd integer): ";
    cin >> kernel_size;
    cout << "Enter sigma value: ";
    cin >> sigma;

    double **kernel = get_Gaussian_Filter(kernel_size, sigma);

    // Apply the Gaussian filter
    Mat filtered_image = convolveGaussianFilter(kernel, kernel_size, stereo_image);

    // Display the original and filtered images
    imshow("Stereo Image", stereo_image);
    imshow("Filtered Image", filtered_image);
    waitKey(0);

    return 0;
}
