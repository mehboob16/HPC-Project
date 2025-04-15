#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01
#define EPOCHS 3
#define BATCH_SIZE 64
#define NUM_CLASSES 10  // Digits 0-9

// Timer function
double get_time(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

// Allocate memory for a matrix
double** allocateMatrix(int rows, int cols) {
    double** mat = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        mat[i] = (double*)malloc(cols * sizeof(double));
    }
    return mat;
}


__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}

void softmax(double* x, int size) {
    double sum = 0;
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i]);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}


typedef struct {
    double* W1;
    double* W2;
    double* b1;
    double* b2;
} NeuralNetworkDevice;
NeuralNetworkDevice host_net;

__global__ void init_W1(double* W1, int input_size, int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = input_size * hidden_size;
    if (idx >= total) return;

    int inputIdx = idx % input_size;
    int hiddenIdx = idx / input_size;

    // Setup random state
    curandState state;
    curand_init(1234, idx, 0, &state);

    double rand_val = curand_uniform_double(&state);
    W1[hiddenIdx * input_size + inputIdx] = (rand_val - 0.5) * 0.1;  // centered around 0
}

__global__ void init_W2(double* W2, int hidden_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = hidden_size * output_size;
    if (idx >= total) return;

    int hiddenIdx = idx % hidden_size;
    int outputIdx = idx / hidden_size;

    // Setup random state
    curandState state;
    curand_init(5678, idx, 0, &state);

    double rand_val = curand_uniform_double(&state);
    W2[outputIdx * hidden_size + hiddenIdx] = (rand_val - 0.5) * 0.1;

}

NeuralNetworkDevice* createNetworkOnDevice(){
    NeuralNetworkDevice* dev_net;

    cudaMalloc((void**)&host_net.W1, sizeof(double) * HIDDEN_SIZE * INPUT_SIZE);
    cudaMalloc((void**)&host_net.W2, sizeof(double) * OUTPUT_SIZE * HIDDEN_SIZE);
    cudaMalloc((void**)&host_net.b1, sizeof(double) * HIDDEN_SIZE);
    cudaMalloc((void**)&host_net.b2, sizeof(double) * OUTPUT_SIZE);
    cudaMemset(host_net.b1, 0, sizeof(double) * HIDDEN_SIZE);
    cudaMemset(host_net.b2, 0, sizeof(double) * OUTPUT_SIZE);
    
    int total_W1 = INPUT_SIZE * HIDDEN_SIZE;
    int total_W2 = HIDDEN_SIZE * OUTPUT_SIZE;

    int threads = 256;
    int blocks_W1 = (total_W1 + threads - 1) / threads;
    int blocks_W2 = (total_W2 + threads - 1) / threads;

    init_W1<<<blocks_W1, threads>>>(host_net.W1, INPUT_SIZE, HIDDEN_SIZE);
    init_W2<<<blocks_W2, threads>>>(host_net.W2, HIDDEN_SIZE, OUTPUT_SIZE);

    cudaMalloc((void**)&dev_net, sizeof(NeuralNetworkDevice));

    cudaDeviceSynchronize();

    cudaMemcpy(dev_net, &host_net, sizeof(NeuralNetworkDevice), cudaMemcpyHostToDevice);

    return dev_net;
}



__global__
void layer1(NeuralNetworkDevice* net, double* input, double* hidden){
    int hiddenIdx = threadIdx.x;

    double hiddenTemp = net->b1[hiddenIdx];
    for (int j = 0; j < INPUT_SIZE; j++)
        hiddenTemp += net->W1[hiddenIdx*INPUT_SIZE + j] * input[j];
    
    hidden[hiddenIdx] = hiddenTemp;
}
__global__
void layer2(NeuralNetworkDevice* net, double* hidden, double* output){
    int outputIdx = threadIdx.x;

    double tempOutput= net->b2[outputIdx];
    for (int j = 0; j < HIDDEN_SIZE; j++)
        tempOutput += net->W2[outputIdx* HIDDEN_SIZE + j] * hidden[j];

    output[outputIdx] = tempOutput;
}

__global__
void relu_d(double* hidden){
    
    double tempHidden = hidden[threadIdx.x];

    tempHidden = tempHidden > 0? tempHidden : 0;

    hidden[threadIdx.x] = tempHidden;
}
// Forward pass
void forward(NeuralNetworkDevice* net, double* input, double* hidden, double* output) {
  
    layer1<<<1, HIDDEN_SIZE>>>(net, input, hidden);

    relu_d<<<1, HIDDEN_SIZE>>>(hidden);

    layer2<<<1, OUTPUT_SIZE>>>(net, hidden, output);
    
    double* tempOutput = (double*)malloc(OUTPUT_SIZE*sizeof(double));
    cudaMemcpy(tempOutput, output, OUTPUT_SIZE*sizeof(double), cudaMemcpyDeviceToHost);

    softmax(tempOutput, OUTPUT_SIZE);

    cudaMemcpy(output, tempOutput, OUTPUT_SIZE*sizeof(double), cudaMemcpyHostToDevice);
}

__global__ 
void layerGradient(double* d_output, double* output, double* target){
    int id = threadIdx.x;

    if(id < OUTPUT_SIZE){
        d_output[id] = output[id]-target[id];
    }
}

__global__
void hiddenLayerGradient(NeuralNetworkDevice* net, double* d_hidden, double* hidden, double* d_output){
    int hiddenIdx = blockIdx.x;
    int outputIdx = threadIdx.x;

    atomicAddDouble(&d_hidden[hiddenIdx], net->W2[outputIdx*HIDDEN_SIZE + hiddenIdx] * d_output[outputIdx]);

    __syncthreads();
    
    if(outputIdx == 0){
        d_hidden[hiddenIdx] *= (hidden[hiddenIdx] > 0);
    }
}

__global__ 
void updateWeights2(NeuralNetworkDevice* net, double* hidden, double*d_output){
    int outputIdx = blockIdx.x;
    int hiddenIdx = threadIdx.x;
    
    int ind = outputIdx * HIDDEN_SIZE + hiddenIdx;

    atomicAddDouble(&net->W2[ind] , -LEARNING_RATE * d_output[outputIdx] * hidden[hiddenIdx]);
    
    if(hiddenIdx == 0)
        net->b2[outputIdx] -= LEARNING_RATE * d_output[outputIdx];
}


__global__
void updateWeights1(NeuralNetworkDevice* net, double* input, double* d_hidden){
    
    int hiddenIdx = blockIdx.x;
    int inputIdx = threadIdx.x;

    int ind = hiddenIdx*INPUT_SIZE + inputIdx;

    atomicAddDouble(&net->W1[ind] ,-LEARNING_RATE * d_hidden[hiddenIdx] * input[inputIdx]);

    if(inputIdx == 0)
        net->b1[hiddenIdx] -= LEARNING_RATE * d_hidden[hiddenIdx];

}

// Backpropagation
void backward(NeuralNetworkDevice* net, double* input, double* hidden, double* output, double* target) {
    // double d_output[OUTPUT_SIZE], d_hidden[HIDDEN_SIZE];
    double* d_output, * d_hidden;
    cudaMalloc((void**)&d_output, OUTPUT_SIZE* sizeof(double));
    cudaMalloc((void**)&d_hidden, HIDDEN_SIZE* sizeof(double));
    cudaMemset(d_output, 0, OUTPUT_SIZE * sizeof(double));
    cudaMemset(d_hidden, 0, HIDDEN_SIZE * sizeof(double));

    layerGradient<<<1, OUTPUT_SIZE>>>(d_output, output, target);

    hiddenLayerGradient<<<HIDDEN_SIZE, OUTPUT_SIZE>>>(net, d_hidden, hidden, d_output);


    updateWeights2<<<OUTPUT_SIZE, HIDDEN_SIZE>>>(net, hidden, d_output);

    updateWeights1<<<HIDDEN_SIZE, INPUT_SIZE>>>(net, input, d_hidden);
}

void assignMemory(double** input, double** hidden, double** output, double** label) {
    cudaMalloc((void**)input, INPUT_SIZE * sizeof(double));
    cudaMalloc((void**)hidden, HIDDEN_SIZE * sizeof(double));
    cudaMalloc((void**)output, OUTPUT_SIZE * sizeof(double));
    cudaMalloc((void**)label, OUTPUT_SIZE * sizeof(double));
}

// Train network
void train(NeuralNetworkDevice* net_d, double** images, double** labels, int numImages) {
    double* input_d, *hidden_d, *output_d, *label_d;
    assignMemory(&input_d, &hidden_d, &output_d, &label_d);

    clock_t total_start = clock();
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double loss = 0.0;
        int correct = 0;

        for (int i = 0; i < numImages; i++) {
            double* output = (double*)malloc(OUTPUT_SIZE * sizeof(double));
            cudaMemcpy(input_d, images[i], INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(label_d, labels[i], OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
            
            forward(net_d, input_d, hidden_d, output_d);
            backward(net_d, input_d, hidden_d, output_d, label_d);
            
            cudaMemcpy(output, output_d, OUTPUT_SIZE* sizeof(double), cudaMemcpyDeviceToHost);

            // Compute loss & accuracy
            for (int k = 0; k < OUTPUT_SIZE; k++) loss -= labels[i][k] * log(output[k] + 1e-12);
            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (output[j] > output[pred]) pred = j;
                if (labels[i][j] > labels[i][actual]) actual = j;
            }
            if (pred == actual) correct++;
        }

        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss / numImages, (correct / (double)numImages) * 100, get_time(epoch_start));
    }
    printf("Total training time: %.3fs\n", get_time(total_start));


}

// Evaluate accuracy on test data
void evaluate(NeuralNetworkDevice* net_d, double** images, double** labels, int numImages) {

    double* input_d, *hidden_d, *output_d, *label_d;
    assignMemory(&input_d, &hidden_d, &output_d, &label_d);

    int correct = 0;
    for (int i = 0; i < numImages; i++) {
        double* output = (double*)malloc(OUTPUT_SIZE * sizeof(double));
        cudaMemcpy(input_d, images[i], INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
            
            


        forward(net_d, input_d, hidden_d, output_d);
        
        cudaMemcpy(output, output_d, OUTPUT_SIZE* sizeof(double), cudaMemcpyDeviceToHost);


        int pred = 0, actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (output[j] > output[pred]) pred = j;
            if (labels[i][j] > labels[i][actual]) actual = j;
        }
        if (pred == actual) correct++;
    }
    printf("Test Accuracy: %.2f%%\n", (correct / (double)numImages) * 100);
}

// Read MNIST dataset
double** loadMNISTImages(const char* filename, int numImages) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 16, SEEK_SET);
    double** images = allocateMatrix(numImages, INPUT_SIZE);
    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            unsigned char pixel;

            // fread(&pixel, sizeof(unsigned char), 1, file);
            if (fread(&pixel, sizeof(unsigned char), 1, file) != 1) {
                fprintf(stderr, "Error: Failed to read pixel\n");
                fclose(file);
                exit(EXIT_FAILURE);
            }

            images[i][j] = pixel / 255.0;
        }
    }
    fclose(file);
    return images;
}


double** loadMNISTLabels(const char* filename, int numLabels) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 8, SEEK_SET);
    double** labels = allocateMatrix(numLabels, OUTPUT_SIZE);
    for (int i = 0; i < numLabels; i++) {
        unsigned char label;
        // fread(&label, sizeof(unsigned char), 1, file);
        if (fread(&label, sizeof(unsigned char), 1, file) != 1) {
            fprintf(stderr, "Error: Failed to read label\n");
            fclose(file);
            exit(EXIT_FAILURE);
        }

        for (int j = 0; j < OUTPUT_SIZE; j++) {
            labels[i][j] = (j == label) ? 1.0 : 0.0;
        }
    }
    fclose(file);
    return labels;
}



// Main function
int main() {
    printf("MNIST Neural Network12\n\n");

    double** train_images = loadMNISTImages("../../data/train-images.idx3-ubyte", 60000);
    double** train_labels = loadMNISTLabels("../../data/train-labels.idx1-ubyte", 60000);
    double** test_images = loadMNISTImages("../../data/t10k-images.idx3-ubyte", 10000);
    double** test_labels = loadMNISTLabels("../../data/t10k-labels.idx1-ubyte", 10000);

    NeuralNetworkDevice* net_d = createNetworkOnDevice();

    train(net_d, train_images, train_labels, 60000);
    evaluate(net_d, test_images, test_labels, 10000);

    return 0;
}

