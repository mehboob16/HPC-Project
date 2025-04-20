#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.007f
#define EPOCHS 3
#define BATCH_SIZE 32
#define NUM_CLASSES 10
#define BLOCK_SIZE 32

__constant__ float input[INPUT_SIZE];

float get_time(clock_t start) {
    return (float)(clock() - start) / CLOCKS_PER_SEC;
}

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float** allocateMatrix(int rows, int cols) {
    float** mat = (float**)malloc(rows * sizeof(float*));
    for (int i = 0; i < rows; i++) {
        mat[i] = (float*)malloc(cols * sizeof(float));
    }
    return mat;
}

typedef struct {
    float* W1;
    float* W2;
    float* b1;
    float* b2;
} NeuralNetworkDevice;
NeuralNetworkDevice host_net;

__global__ void init_W1(float* W1) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = INPUT_SIZE * HIDDEN_SIZE;
    if (idx >= total) return;

    int inputIdx = idx % INPUT_SIZE;
    int hiddenIdx = idx / INPUT_SIZE;

    curandState state;
    curand_init(1234, idx, 0, &state);

    float rand_val = curand_uniform(&state);
    W1[hiddenIdx * INPUT_SIZE + inputIdx] = (rand_val - 0.5f) * 0.1f;
}

__global__ void init_W2(float* W2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = HIDDEN_SIZE * OUTPUT_SIZE;
    if (idx >= total) return;

    int hiddenIdx = idx % HIDDEN_SIZE;
    int outputIdx = idx / HIDDEN_SIZE;

    curandState state;
    curand_init(5678, idx, 0, &state);

    float rand_val = curand_uniform(&state);
    W2[outputIdx * HIDDEN_SIZE + hiddenIdx] = (rand_val - 0.5f) * 0.1f;
}

NeuralNetworkDevice* createNetworkOnDevice() {
    NeuralNetworkDevice* dev_net;

    cudaMalloc((void**)&host_net.W1, sizeof(float) * HIDDEN_SIZE * INPUT_SIZE);
    cudaMalloc((void**)&host_net.W2, sizeof(float) * OUTPUT_SIZE * HIDDEN_SIZE);
    cudaMalloc((void**)&host_net.b1, sizeof(float) * HIDDEN_SIZE);
    cudaMalloc((void**)&host_net.b2, sizeof(float) * OUTPUT_SIZE);
    cudaMemset(host_net.b1, 0, sizeof(float) * HIDDEN_SIZE);
    cudaMemset(host_net.b2, 0, sizeof(float) * OUTPUT_SIZE);

    int total_W1 = INPUT_SIZE * HIDDEN_SIZE;
    int total_W2 = HIDDEN_SIZE * OUTPUT_SIZE;

    int threads = 256;
    int blocks_W1 = (total_W1 + threads - 1) / threads;
    int blocks_W2 = (total_W2 + threads - 1) / threads;

    init_W1<<<blocks_W1, threads>>>(host_net.W1);
    init_W2<<<blocks_W2, threads>>>(host_net.W2);

    cudaMalloc((void**)&dev_net, sizeof(NeuralNetworkDevice));
    cudaMemcpy(dev_net, &host_net, sizeof(NeuralNetworkDevice), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    return dev_net;
}


__global__ void init_hidden(NeuralNetworkDevice* net, float* hidden) {
    int idx = threadIdx.x;
    if (idx < HIDDEN_SIZE) {
        hidden[idx] = net->b1[idx];
    }
}

__global__ void hiddenLayerForwardBatch(NeuralNetworkDevice* net, float* input_batch, float* hidden_batch) {
    extern __shared__ float input_shared[];

    int batch_idx = blockIdx.x;
    int neuron_idx = threadIdx.x;

    if (neuron_idx < INPUT_SIZE) {
        input_shared[neuron_idx] = input_batch[batch_idx * INPUT_SIZE + neuron_idx];
    }
    __syncthreads();

    if (neuron_idx < HIDDEN_SIZE) {
        float sum = net->b1[neuron_idx];
        for (int j = 0; j < INPUT_SIZE; j++) {
            sum += input_shared[j] * net->W1[neuron_idx * INPUT_SIZE + j];
        }
        hidden_batch[batch_idx * HIDDEN_SIZE + neuron_idx] = sum;
    }
}

__global__ void relu_d(float* hidden) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < BATCH_SIZE * HIDDEN_SIZE && hidden[idx] < 0.0f)
        hidden[idx] = 0.0f;
}

__global__ void outputLayerForwardBatch(NeuralNetworkDevice* net, float* hidden_batch, float* output_batch) {
    extern __shared__ float hidden_shared[];

    int batch_idx = blockIdx.x;
    int neuron_idx = threadIdx.x;

    if (neuron_idx < HIDDEN_SIZE) {
        hidden_shared[neuron_idx] = hidden_batch[batch_idx * HIDDEN_SIZE + neuron_idx];
    }
    __syncthreads();

    if (neuron_idx < OUTPUT_SIZE) {
        float sum = net->b2[neuron_idx];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            sum += hidden_shared[j] * net->W2[neuron_idx * HIDDEN_SIZE + j];
        }
        output_batch[batch_idx * OUTPUT_SIZE + neuron_idx] = sum;
    }
}

__global__ void softmax_batch(float* output_batch) {
    int batch_idx = blockIdx.x;

    float max_val = -1e20f;
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        float val = output_batch[batch_idx * OUTPUT_SIZE + i];
        if (val > max_val) max_val = val;
    }

    float sum = 0.0f;
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        float val = expf(output_batch[batch_idx * OUTPUT_SIZE + i] - max_val);
        output_batch[batch_idx * OUTPUT_SIZE + i] = val;
        sum += val;
    }

    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        output_batch[batch_idx * OUTPUT_SIZE + i] /= sum;
    }
}

void forward(NeuralNetworkDevice* net, float* hidden_batch, float* output_batch, float* input_batch, cudaStream_t stream) {
    size_t shared_input = INPUT_SIZE * sizeof(float);
    hiddenLayerForwardBatch<<<BATCH_SIZE, max(HIDDEN_SIZE, INPUT_SIZE), shared_input, stream>>>(net, input_batch, hidden_batch);

    int reluThreads = 256;
    int reluBlocks = (BATCH_SIZE * HIDDEN_SIZE + reluThreads - 1) / reluThreads;
    relu_d<<<reluBlocks, reluThreads, 0, stream>>>(hidden_batch);

    size_t shared_hidden = HIDDEN_SIZE * sizeof(float);
    outputLayerForwardBatch<<<BATCH_SIZE, max(HIDDEN_SIZE, OUTPUT_SIZE), shared_hidden, stream>>>(net, hidden_batch, output_batch);

    softmax_batch<<<BATCH_SIZE, 1, 0, stream>>>(output_batch);
}

__global__
void layerGradientBatch(float* d_output_batch, float* output_batch, float* target_batch) {
    int batch_idx = blockIdx.x;
    int i = threadIdx.x;

    int offset = batch_idx * OUTPUT_SIZE + i;
    if (i < OUTPUT_SIZE)
        d_output_batch[offset] = output_batch[offset] - target_batch[offset];
}


__global__
void hiddenLayerGradientBatch(NeuralNetworkDevice* net, float* d_hidden_batch, float* hidden_batch, float* d_output_batch) {
    int batch_idx = blockIdx.x;
    int hidden_idx = threadIdx.x;

    float sum = 0.0f;
    for (int output_idx = 0; output_idx < OUTPUT_SIZE; ++output_idx) {
        int output_offset = batch_idx * OUTPUT_SIZE + output_idx;
        int w_idx = output_idx * HIDDEN_SIZE + hidden_idx;
        sum += net->W2[w_idx] * d_output_batch[output_offset];
    }

    int h_off = batch_idx * HIDDEN_SIZE + hidden_idx;
    float relu_grad = (hidden_batch[h_off] > 0.0f) ? 1.0f : 0.0f;
    d_hidden_batch[h_off] = sum * relu_grad;
}

__global__
void updateWeights1Batch(NeuralNetworkDevice* net, float* d_hidden_batch, float* input_batch, int batch_size) {
    int hidden_idx = blockIdx.x;
    int input_idx = threadIdx.x;

    float grad = 0.0f;
    float bias_grad = 0.0f;

    for (int b = 0; b < batch_size; b++) {
        int h_off = b * HIDDEN_SIZE + hidden_idx;
        int i_off = b * INPUT_SIZE + input_idx;
        grad += d_hidden_batch[h_off] * input_batch[i_off];
        if (input_idx == 0) bias_grad += d_hidden_batch[h_off];
    }

    int w_idx = hidden_idx * INPUT_SIZE + input_idx;
    atomicAdd(&net->W1[w_idx], -LEARNING_RATE * grad / batch_size);

    if (input_idx == 0) {
        atomicAdd(&net->b1[hidden_idx], -LEARNING_RATE * bias_grad / batch_size);
    }
}

__global__
void updateWeights2Batch(NeuralNetworkDevice* net, float* hidden_batch, float* d_output_batch, int batch_size) {
    int output_idx = blockIdx.x;
    int hidden_idx = threadIdx.x;

    float grad = 0.0f;
    float bias_grad = 0.0f;

    for (int b = 0; b < batch_size; b++) {
        int h_off = b * HIDDEN_SIZE + hidden_idx;
        int o_off = b * OUTPUT_SIZE + output_idx;
        grad += d_output_batch[o_off] * hidden_batch[h_off];
        if (hidden_idx == 0) bias_grad += d_output_batch[o_off];
    }

    int w_idx = output_idx * HIDDEN_SIZE + hidden_idx;
    atomicAdd(&net->W2[w_idx], -LEARNING_RATE * grad / batch_size);

    if (hidden_idx == 0) {
        atomicAdd(&net->b2[output_idx], -LEARNING_RATE * bias_grad / batch_size);
    }
}

void backward_batch(NeuralNetworkDevice* net, float* hidden_batch, float* output_batch, float* label_batch,
                    float* input_batch, int batch_size, cudaStream_t stream) {
    float* d_output_batch;
    float* d_hidden_batch;

    cudaMallocAsync(&d_output_batch, batch_size * OUTPUT_SIZE * sizeof(float), stream);
    cudaMallocAsync(&d_hidden_batch, batch_size * HIDDEN_SIZE * sizeof(float), stream);

    cudaMemsetAsync(d_output_batch, 0, batch_size * OUTPUT_SIZE * sizeof(float), stream);
    cudaMemsetAsync(d_hidden_batch, 0, batch_size * HIDDEN_SIZE * sizeof(float), stream);

    dim3 grid1(batch_size);
    dim3 block1(OUTPUT_SIZE);
    layerGradientBatch<<<grid1, block1, 0, stream>>>(d_output_batch, output_batch, label_batch);

    dim3 grid2(batch_size);
    dim3 block2(HIDDEN_SIZE);
    hiddenLayerGradientBatch<<<grid2, block2, 0, stream>>>(net, d_hidden_batch, hidden_batch, d_output_batch);

    dim3 grid3(OUTPUT_SIZE);
    dim3 block3(HIDDEN_SIZE);
    updateWeights2Batch<<<grid3, block3, 0, stream>>>(net, hidden_batch, d_output_batch, batch_size);

    dim3 grid4(HIDDEN_SIZE);
    dim3 block4(INPUT_SIZE);
    updateWeights1Batch<<<grid4, block4, 0, stream>>>(net, d_hidden_batch, input_batch, batch_size);

    cudaFreeAsync(d_output_batch, stream);
    cudaFreeAsync(d_hidden_batch, stream);
}

void assignMemory(float** hidden, float** output, float** label) {
    cudaMalloc((void**)hidden, HIDDEN_SIZE * sizeof(float));
    cudaMalloc((void**)output, OUTPUT_SIZE * sizeof(float));
    cudaMalloc((void**)label, OUTPUT_SIZE * sizeof(float));
}


void train(NeuralNetworkDevice* net_d, float** images, float** labels, int numImages) {
    size_t hidden_bytes = BATCH_SIZE * HIDDEN_SIZE * sizeof(float);
    size_t input_bytes  = BATCH_SIZE * INPUT_SIZE * sizeof(float);
    size_t output_bytes = BATCH_SIZE * OUTPUT_SIZE * sizeof(float);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    float *input_batch_d, *hidden_d, *output_d, *label_batch_d;
    cudaMallocAsync(&input_batch_d, input_bytes, stream);
    cudaMallocAsync(&output_d, output_bytes, stream);
    cudaMallocAsync(&label_batch_d, output_bytes, stream);
    cudaMallocAsync(&hidden_d, hidden_bytes, stream);

    float* input_host;
    float* label_host;
    float* output_host;
    cudaMallocHost(&input_host, input_bytes);
    cudaMallocHost(&label_host, output_bytes);
    cudaMallocHost(&output_host, output_bytes);

    clock_t total_start = clock();

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        float loss = 0.0f;
        int correct = 0;

        for (int i = 0; i < numImages; i += BATCH_SIZE) {
            int current_batch_size = min(BATCH_SIZE, numImages - i);

            for (int b = 0; b < current_batch_size; b++) {
                memcpy(input_host + b * INPUT_SIZE, images[i + b], INPUT_SIZE * sizeof(float));
                memcpy(label_host + b * OUTPUT_SIZE, labels[i + b], OUTPUT_SIZE * sizeof(float));
            }

            cudaMemcpyAsync(input_batch_d, input_host, current_batch_size * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(label_batch_d, label_host, current_batch_size * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream);

            forward(net_d, hidden_d, output_d, input_batch_d, stream);
            backward_batch(net_d, hidden_d, output_d, label_batch_d, input_batch_d, current_batch_size, stream);

            cudaStreamSynchronize(stream);

            cudaMemcpy(output_host, output_d, current_batch_size * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

            for (int b = 0; b < current_batch_size; b++) {
                int idx = i + b;
                float* output_row = output_host + b * OUTPUT_SIZE;
                float* label_row = labels[idx];

                for (int k = 0; k < OUTPUT_SIZE; k++)
                    loss -= label_row[k] * logf(output_row[k] + 1e-12f);

                int pred = 0, actual = 0;
                for (int j = 0; j < OUTPUT_SIZE; j++) {
                    if (output_row[j] > output_row[pred]) pred = j;
                    if (label_row[j] > label_row[actual]) actual = j;
                }
                if (pred == actual) correct++;
            }
        }

        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss / numImages, (correct / (float)numImages) * 100, get_time(epoch_start));
    }

    printf("Total training time: %.3fs\n", get_time(total_start));

    cudaStreamDestroy(stream);
    cudaFreeAsync(input_batch_d, stream);
    cudaFreeAsync(hidden_d, stream);
    cudaFreeAsync(output_d, stream);
    cudaFreeAsync(label_batch_d, stream);

    cudaFreeHost(input_host);
    cudaFreeHost(label_host);
    cudaFreeHost(output_host);
}

void evaluate(NeuralNetworkDevice* net_d, float** images, float** labels, int numImages) {
    size_t input_bytes = INPUT_SIZE * sizeof(float);
    size_t hidden_bytes = HIDDEN_SIZE * sizeof(float);
    size_t output_bytes = OUTPUT_SIZE * sizeof(float);

    cudaStream_t eval_stream;
    cudaStreamCreate(&eval_stream);

    float *input_batch_d, *hidden_d, *output_d, *label_d;
    cudaMallocAsync(&input_batch_d, input_bytes, eval_stream);
    cudaMallocAsync(&hidden_d, hidden_bytes, eval_stream);
    cudaMallocAsync(&output_d, output_bytes, eval_stream);
    cudaMallocAsync(&label_d, output_bytes, eval_stream);

    float* output_host;
    cudaMallocHost(&output_host, output_bytes);

    int correct = 0;

    for (int i = 0; i < numImages; i++) {
        cudaMemcpyAsync(input_batch_d, images[i], input_bytes, cudaMemcpyHostToDevice, eval_stream);
        forward(net_d, hidden_d, output_d, input_batch_d, eval_stream);
        cudaStreamSynchronize(eval_stream);
        cudaMemcpy(output_host, output_d, output_bytes, cudaMemcpyDeviceToHost);

        int pred = 0, actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (output_host[j] > output_host[pred]) pred = j;
            if (labels[i][j] > labels[i][actual]) actual = j;
        }
        if (pred == actual) correct++;
    }

    printf("Test Accuracy: %.2f%%\n", (correct / (float)numImages) * 100.0f);

    cudaFreeAsync(input_batch_d, eval_stream);
    cudaFreeAsync(hidden_d, eval_stream);
    cudaFreeAsync(output_d, eval_stream);
    cudaFreeAsync(label_d, eval_stream);
    cudaFreeHost(output_host);
    cudaStreamDestroy(eval_stream);
}


float** loadMNISTImages(const char* filename, int numImages) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 16, SEEK_SET);
    float** images = allocateMatrix(numImages, INPUT_SIZE);
    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            unsigned char pixel;
            if (fread(&pixel, sizeof(unsigned char), 1, file) != 1) {
                fprintf(stderr, "Error: Failed to read pixel\n");
                fclose(file);
                exit(EXIT_FAILURE);
            }
            images[i][j] = pixel / 255.0f;
        }
    }
    fclose(file);
    return images;
}

float** loadMNISTLabels(const char* filename, int numLabels) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 8, SEEK_SET);
    float** labels = allocateMatrix(numLabels, OUTPUT_SIZE);
    for (int i = 0; i < numLabels; i++) {
        unsigned char label;
        if (fread(&label, sizeof(unsigned char), 1, file) != 1) {
            fprintf(stderr, "Error: Failed to read label\n");
            fclose(file);
            exit(EXIT_FAILURE);
        }
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            labels[i][j] = (j == label) ? 1.0f : 0.0f;
        }
    }
    fclose(file);
    return labels;
}

int main() {
    printf("MNIST Neural Network with Tensor Cores\n\n");

    float** train_images = loadMNISTImages("../../data/train-images.idx3-ubyte", 60000);
    float** train_labels = loadMNISTLabels("../../data/train-labels.idx1-ubyte", 60000);
    float** test_images = loadMNISTImages("../../data/t10k-images.idx3-ubyte", 10000);
    float** test_labels = loadMNISTLabels("../../data/t10k-labels.idx1-ubyte", 10000);
    NeuralNetworkDevice* net_d = createNetworkOnDevice();

    train(net_d, train_images, train_labels, 60000);
    evaluate(net_d, test_images, test_labels, 10000);

    return 0;
}
