#include <nppi.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>

// CUDA error-checking macro
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// Dummy MNIST model weights (10 classes, flattened 28x28 image size)
constexpr int INPUT_SIZE = 28 * 28;
constexpr int NUM_CLASSES = 10;
__constant__ float d_weights[INPUT_SIZE * NUM_CLASSES];
__constant__ float d_biases[NUM_CLASSES];

// Kernel to classify a batch of MNIST images
__global__ void classifyMNIST(const float* d_images, int batch_size, int* d_labels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const float* image = d_images + idx * INPUT_SIZE;
    float scores[NUM_CLASSES] = {0};

    for (int i = 0; i < NUM_CLASSES; ++i) {
        for (int j = 0; j < INPUT_SIZE; ++j) {
            scores[i] += image[j] * d_weights[i * INPUT_SIZE + j];
        }
        scores[i] += d_biases[i];
    }

    // Find the class with the highest score
    int best_class = 0;
    float best_score = scores[0];
    for (int i = 1; i < NUM_CLASSES; ++i) {
        if (scores[i] > best_score) {
            best_score = scores[i];
            best_class = i;
        }
    }

    d_labels[idx] = best_class;
}

// Preprocess a single image using CUDA NPP
void preprocessImage(const cv::Mat& inputImage, float* d_output) {
    int width = inputImage.cols;
    int height = inputImage.rows;

    // Allocate device memory
    Npp8u *d_input;
    size_t pitch;
    CUDA_CHECK(cudaMallocPitch(&d_input, &pitch, width * sizeof(Npp8u), height));

    // Copy input image to device memory
    CUDA_CHECK(cudaMemcpy2D(d_input, pitch, inputImage.data, width * sizeof(Npp8u), width * sizeof(Npp8u), height, cudaMemcpyHostToDevice));

    // Define ROI
    NppiSize roi = {width, height};

    // Normalize the image to [0, 1] (MNIST images are grayscale)
    NPP_CHECK(nppiScale_8u32f_C1R(
        d_input, pitch, d_output, width * sizeof(float), roi, 0.0f, 1.0f));

    // Free device memory
    CUDA_CHECK(cudaFree(d_input));
}

// Main function
int main(int argc, char** argv) {
    // Check arguments
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <mnist_image_path>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string imagePath = argv[1];

    // Load and preprocess the image
    cv::Mat inputImage = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (inputImage.empty() || inputImage.cols != 28 || inputImage.rows != 28) {
        std::cerr << "Error: Input image must be a 28x28 grayscale image." << std::endl;
        return EXIT_FAILURE;
    }

    // Allocate device memory for preprocessed image
    float* d_image;
    CUDA_CHECK(cudaMalloc(&d_image, INPUT_SIZE * sizeof(float)));

    preprocessImage(inputImage, d_image);

    // Allocate memory for classification output
    int* d_label;
    int h_label;
    CUDA_CHECK(cudaMalloc(&d_label, sizeof(int)));

    // Perform classification
    classifyMNIST<<<1, 1>>>(d_image, 1, d_label);

    // Copy label back to host
    CUDA_CHECK(cudaMemcpy(&h_label, d_label, sizeof(int), cudaMemcpyDeviceToHost));

    std::cout << "Predicted label: " << h_label << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFree(d_image));
    CUDA_CHECK(cudaFree(d_label));

    return 0;
}
