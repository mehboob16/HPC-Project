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
#define BATCH_SIZE 16
#define NUM_CLASSES 10  // Digits 0-9
#define BLOCK_SIZE 32

__constant__  double input[INPUT_SIZE];

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

__global__ void init_W1(double* W1) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = INPUT_SIZE * HIDDEN_SIZE;
    if (idx >= total) return;

    int inputIdx = idx % INPUT_SIZE;
    int hiddenIdx = idx / INPUT_SIZE;

    // Setup random state
    curandState state;
    curand_init(1234, idx, 0, &state);

    double rand_val = curand_uniform_double(&state);
    W1[hiddenIdx * INPUT_SIZE + inputIdx] = (rand_val - 0.5) * 0.1;  // centered around 0
}

__global__ void init_W2(double* W2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = HIDDEN_SIZE * OUTPUT_SIZE;
    if (idx >= total) return;

    int hiddenIdx = idx % HIDDEN_SIZE;
    int outputIdx = idx / HIDDEN_SIZE;

    // Setup random state
    curandState state;
    curand_init(5678, idx, 0, &state);

    double rand_val = curand_uniform_double(&state);
    W2[outputIdx * HIDDEN_SIZE + hiddenIdx] = (rand_val - 0.5) * 0.1;

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

    init_W1<<<blocks_W1, threads>>>(host_net.W1);
    init_W2<<<blocks_W2, threads>>>(host_net.W2);

    cudaMalloc((void**)&dev_net, sizeof(NeuralNetworkDevice));

    cudaDeviceSynchronize();

    cudaMemcpy(dev_net, &host_net, sizeof(NeuralNetworkDevice ), cudaMemcpyHostToDevice);

    return dev_net;
}


__global__ void init_hidden(NeuralNetworkDevice* net, double* hidden){
    hidden[threadIdx.x] = net->b1[threadIdx.x];

}
__global__
void layer1(NeuralNetworkDevice* net, double* hidden){
    int hiddenIdx = blockIdx.x*blockDim.x + threadIdx.x;
    int inputIdx = blockIdx.y*blockDim.y + threadIdx.y;
    int hiddenLocal = threadIdx.x;
    int inputLocal = threadIdx.y;

    if(hiddenIdx < HIDDEN_SIZE && inputIdx < INPUT_SIZE ){
        __shared__ double sInput[BLOCK_SIZE];
        __shared__ double sHidden[BLOCK_SIZE];

        if(inputLocal == 0)
            sHidden[hiddenLocal] = 0;
        
        if(hiddenLocal == 0)
            sInput[inputLocal] = input[inputIdx];

        __syncthreads();
 
        atomicAddDouble(&sHidden[hiddenLocal], net->W1[hiddenIdx*INPUT_SIZE + inputIdx] * sInput[inputLocal]);

        __syncthreads();

        if(inputLocal == 0)
            atomicAddDouble(&hidden[hiddenIdx], sHidden[hiddenLocal]);
    }
}
__global__
void layer2(NeuralNetworkDevice* net, double* hidden, double* output) {
    int outputIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int hiddenIdx = blockIdx.y * blockDim.y + threadIdx.y;
    int outputLocal = threadIdx.x;
    int hiddenLocal = threadIdx.y;

    if (outputIdx < OUTPUT_SIZE && hiddenIdx < HIDDEN_SIZE) {
        __shared__ double sHidden[BLOCK_SIZE];
        __shared__ double sOutput[BLOCK_SIZE];

        if (hiddenLocal == 0)
            sOutput[outputLocal] = 0;

        if (outputLocal == 0)
            sHidden[hiddenLocal] = hidden[hiddenIdx];

        __syncthreads();

        atomicAddDouble(&sOutput[outputLocal], net->W2[outputIdx * HIDDEN_SIZE + hiddenIdx] * sHidden[hiddenLocal]);

        __syncthreads();

        if (hiddenLocal == 0)
            atomicAddDouble(&output[outputIdx], sOutput[outputLocal]);
    }
}

__global__
void relu_d(double* hidden){
    
    double tempHidden = hidden[threadIdx.x];

    tempHidden = tempHidden > 0? tempHidden : 0;

    hidden[threadIdx.x] = tempHidden;
}

// Forward pass
void forward_stream(NeuralNetworkDevice* net, double* hidden, double* output, double* input_d, cudaStream_t stream) {
    cudaMemcpyToSymbolAsync(input, input_d, INPUT_SIZE * sizeof(double), 0, cudaMemcpyDeviceToDevice, stream);


    init_hidden<<<1, HIDDEN_SIZE, 0, stream>>>(net, hidden);
    cudaDeviceSynchronize();

    dim3 block((HIDDEN_SIZE + BLOCK_SIZE - 1)/BLOCK_SIZE, (INPUT_SIZE + BLOCK_SIZE - 1)/BLOCK_SIZE+1);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    layer1<<<block, threads, 0, stream>>>(net, hidden);

    relu_d<<<1, HIDDEN_SIZE, 0, stream>>>(hidden);

    dim3 block2((OUTPUT_SIZE + BLOCK_SIZE - 1)/BLOCK_SIZE, (HIDDEN_SIZE + BLOCK_SIZE - 1)/BLOCK_SIZE);
    dim3 threads2(BLOCK_SIZE, BLOCK_SIZE);
    layer2<<<block2, threads2, 0, stream>>>(net, hidden, output);
    // layer2<<<1, OUTPUT_SIZE>>>(net, hidden, output);
    
    double* tempOutput = (double*)malloc(OUTPUT_SIZE*sizeof(double));
    cudaMemcpyAsync(tempOutput, output, OUTPUT_SIZE*sizeof(double), cudaMemcpyDeviceToHost, stream);

    softmax(tempOutput, OUTPUT_SIZE);

    cudaMemcpyAsync(output, tempOutput, OUTPUT_SIZE*sizeof(double), cudaMemcpyHostToDevice, stream );
}
void forward(NeuralNetworkDevice* net, double* hidden, double* output) {

    init_hidden<<<1, HIDDEN_SIZE>>>(net, hidden);
    cudaDeviceSynchronize();

    dim3 block((HIDDEN_SIZE + BLOCK_SIZE - 1)/BLOCK_SIZE, (INPUT_SIZE + BLOCK_SIZE - 1)/BLOCK_SIZE+1);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    layer1<<<block, threads>>>(net, hidden);

    relu_d<<<1, HIDDEN_SIZE>>>(hidden);

    dim3 block2((OUTPUT_SIZE + BLOCK_SIZE - 1)/BLOCK_SIZE, (HIDDEN_SIZE + BLOCK_SIZE - 1)/BLOCK_SIZE);
    dim3 threads2(BLOCK_SIZE, BLOCK_SIZE);
    layer2<<<block2, threads2>>>(net, hidden, output);
    // layer2<<<1, OUTPUT_SIZE>>>(net, hidden, output);
    
    double* tempOutput = (double*)malloc(OUTPUT_SIZE*sizeof(double));
    cudaMemcpy(tempOutput, output, OUTPUT_SIZE*sizeof(double), cudaMemcpyDeviceToHost);

    softmax(tempOutput, OUTPUT_SIZE);

    cudaMemcpy(output, tempOutput, OUTPUT_SIZE*sizeof(double), cudaMemcpyHostToDevice);
}

__global__ 
void layerGradient(double* d_output, double* output, double* target){
    int id = threadIdx.x;

    if(id < OUTPUT_SIZE){
    for(int id = 0; id < OUTPUT_SIZE; id++)
        d_output[id] = output[id]-target[id];
    }
}

__global__
void hiddenLayerGradient(NeuralNetworkDevice* net, double* d_hidden, double* hidden, double* d_output) {
    int hiddenIdx = threadIdx.x + blockIdx.x * blockDim.x;

    if (hiddenIdx >= HIDDEN_SIZE) return;

    __shared__ double shared_output[OUTPUT_SIZE];

    int tid = threadIdx.x;
    for (int i = tid; i < OUTPUT_SIZE; i += blockDim.x) {
        shared_output[i] = d_output[i];
    }

    __syncthreads();

    double grad = 0.0;
    for (int outputIdx = 0; outputIdx < OUTPUT_SIZE; outputIdx++) {
        double w = net->W2[outputIdx * HIDDEN_SIZE + hiddenIdx]; 
        grad += w * shared_output[outputIdx];
    }

    d_hidden[hiddenIdx] = grad * (hidden[hiddenIdx] > 0);
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
void updateWeights1(NeuralNetworkDevice* net, double* d_hidden){
    
    int hiddenIdx = blockIdx.x;
    int inputIdx = threadIdx.x;

    int ind = hiddenIdx*INPUT_SIZE + inputIdx;

    atomicAddDouble(&net->W1[ind] ,-LEARNING_RATE * d_hidden[hiddenIdx] * input[inputIdx]);

    if(inputIdx == 0)
        net->b1[hiddenIdx] -= LEARNING_RATE * d_hidden[hiddenIdx];

}

// Backpropagation
void backward(NeuralNetworkDevice* net, double* hidden, double* output, double* target, cudaStream_t stream) {
    double* d_output;
    double* d_hidden;

    cudaMalloc(&d_output, OUTPUT_SIZE * sizeof(double));
    cudaMalloc(&d_hidden, HIDDEN_SIZE * sizeof(double));
    cudaMemset(d_output, 0, OUTPUT_SIZE * sizeof(double));
    cudaMemset(d_hidden, 0, HIDDEN_SIZE * sizeof(double));

    layerGradient<<<1, OUTPUT_SIZE>>>(d_output, output, target);
    hiddenLayerGradient<<<1, HIDDEN_SIZE>>>(net, d_hidden, hidden, d_output);

    updateWeights2<<<OUTPUT_SIZE, HIDDEN_SIZE>>>(net, hidden, d_output);
    updateWeights1<<<HIDDEN_SIZE, INPUT_SIZE>>>(net, d_hidden);
    
}

void backward_stream(NeuralNetworkDevice* net, double* hidden, double* output, double* target, cudaStream_t stream) {
    double* d_output;
    double* d_hidden;

    cudaMallocAsync(&d_output, OUTPUT_SIZE * sizeof(double), stream);
    cudaMallocAsync(&d_hidden, HIDDEN_SIZE * sizeof(double), stream);
    cudaMemset(d_output, 0, OUTPUT_SIZE * sizeof(double));
    cudaMemset(d_hidden, 0, HIDDEN_SIZE * sizeof(double));

    layerGradient<<<1, OUTPUT_SIZE, 0, stream>>>(d_output, output, target);
    hiddenLayerGradient<<<1, HIDDEN_SIZE, 0, stream>>>(net, d_hidden, hidden, d_output);

    updateWeights2<<<OUTPUT_SIZE, HIDDEN_SIZE, 0, stream>>>(net, hidden, d_output);
    updateWeights1<<<HIDDEN_SIZE, INPUT_SIZE, 0, stream>>>(net, d_hidden);
    
    cudaFreeAsync(d_output, stream);
    cudaFreeAsync(d_hidden, stream);
}

void assignMemory( double** hidden, double** output, double** label) {
    cudaMalloc((void**)hidden, HIDDEN_SIZE * sizeof(double));
    cudaMalloc((void**)output, OUTPUT_SIZE * sizeof(double));
    cudaMalloc((void**)label, OUTPUT_SIZE * sizeof(double));
}

// Train network
void train(NeuralNetworkDevice* net_d, double** images, double** labels, int numImages) {

    const int NUM_STREAMS = 4;
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // double *hidden_d, *output_d, *label_d;
    // assignMemory(&hidden_d, &output_d, &label_d);
    double *input_d[NUM_STREAMS], *label_d[NUM_STREAMS], *output_d[NUM_STREAMS], *hidden_d[NUM_STREAMS];
    double* h_output[NUM_STREAMS];

    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaMalloc(&input_d[i], INPUT_SIZE * sizeof(double));
        cudaMalloc(&label_d[i], OUTPUT_SIZE * sizeof(double));
        cudaMalloc(&output_d[i], OUTPUT_SIZE * sizeof(double));
        cudaMalloc(&hidden_d[i], HIDDEN_SIZE * sizeof(double));
        cudaHostAlloc(&h_output[i], OUTPUT_SIZE * sizeof(double), cudaHostAllocDefault);
    }


    clock_t total_start = clock();
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double loss = 0.0;
        int correct = 0;

        for (int i = 0; i < numImages; i += NUM_STREAMS) {
            for (int s = 0; s < NUM_STREAMS && (i + s) < numImages; s++) {
                int imgIdx = i + s;
        
                cudaMemcpyAsync(input_d[s], images[imgIdx], INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice, streams[s]);
                cudaMemcpyAsync(label_d[s], labels[imgIdx], OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice, streams[s]);
        
                // Forward/Backward kernels — need to support stream as param
                forward_stream(net_d, hidden_d[s], output_d[s], input_d[s], streams[s]);
                backward_stream(net_d, hidden_d[s], output_d[s], label_d[s], streams[s]);
        
                // Copy output back
                cudaMemcpyAsync(h_output[s], output_d[s], OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost, streams[s]);
            }
        
            // Sync all streams before processing outputs
            for (int s = 0; s < NUM_STREAMS && (i + s) < numImages; s++) {
                cudaStreamSynchronize(streams[s]);
        
                // Evaluate result
                int pred = 0, actual = 0;
                for (int j = 0; j < OUTPUT_SIZE; j++) {
                    if (h_output[s][j] > h_output[s][pred]) pred = j;
                    if (labels[i + s][j] > labels[i + s][actual]) actual = j;
                    loss -= labels[i + s][j] * log(h_output[s][j] + 1e-12);
                }
                if (pred == actual) correct++;
            }
        }
        
        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss / numImages, (correct / (double)numImages) * 100, get_time(epoch_start));
    }
    printf("Total training time: %.3fs\n", get_time(total_start));


    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaFree(input_d[i]);
        cudaFree(label_d[i]);
        cudaFree(output_d[i]);
        cudaFree(hidden_d[i]);
        cudaFreeHost(h_output[i]);
        cudaStreamDestroy(streams[i]);
    }
    
}

// Evaluate accuracy on test data
void evaluate(NeuralNetworkDevice* net_d, double** images, double** labels, int numImages) {

    double *hidden_d, *output_d, *label_d;
    assignMemory(&hidden_d, &output_d, &label_d);

    int correct = 0;
    for (int i = 0; i < numImages; i++) {
        double* output = (double*)malloc(OUTPUT_SIZE * sizeof(double));
        cudaMemcpyToSymbol(input, images[i], INPUT_SIZE * sizeof(double));
  
        forward(net_d,  hidden_d, output_d);
        
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
    printf("MNIST Neural Network3\n\n");

    double** train_images = loadMNISTImages("../data/train-images.idx3-ubyte", 60000);
    double** train_labels = loadMNISTLabels("../data/train-labels.idx1-ubyte", 60000);
    double** test_images = loadMNISTImages("../data/t10k-images.idx3-ubyte", 10000);
    double** test_labels = loadMNISTLabels("../data/t10k-labels.idx1-ubyte", 10000);
    NeuralNetworkDevice* net_d = createNetworkOnDevice();

    train(net_d, train_images, train_labels, 60000);
    evaluate(net_d, test_images, test_labels, 10000);

    return 0;
}

