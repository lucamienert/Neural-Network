#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INPUT_SIZE 3
#define HIDDEN_SIZE 4
#define OUTPUT_SIZE 2
#define LEARNING_RATE 0.1
#define EPOCHS 1000

typedef struct 
{
    double *inputs;
    double *weights;
    double bias;
    double output;
    double error;
} Neuron;

typedef struct 
{
    int num_inputs;
    Neuron *neurons;
} Layer;

typedef struct 
{
    int num_layers;
    Layer *layers;
} NeuralNetwork;

double sigmoid(double x) 
{
    return 1 / (1 + exp(-x));
}

double sigmoid_derivative(double x) 
{
    return x * (1 - x);
}

NeuralNetwork initialize_network() 
{
    NeuralNetwork network;
    network.num_layers = 2;

    network.layers = (Layer *)malloc(network.num_layers * sizeof(Layer));

    network.layers[0].num_inputs = INPUT_SIZE;
    network.layers[0].neurons = (Neuron *)malloc(INPUT_SIZE * sizeof(Neuron));

    for (int i = 0; i < INPUT_SIZE; i++) 
    {
        network.layers[0].neurons[i].inputs = NULL;
        network.layers[0].neurons[i].weights = NULL;
    }

    network.layers[1].num_inputs = HIDDEN_SIZE;
    network.layers[1].neurons = (Neuron *)malloc(HIDDEN_SIZE * sizeof(Neuron));
    for (int i = 0; i < HIDDEN_SIZE; i++) 
    {
        network.layers[1].neurons[i].inputs = (double *)malloc(INPUT_SIZE * sizeof(double));
        network.layers[1].neurons[i].weights = (double *)malloc(INPUT_SIZE * sizeof(double));

        for (int j = 0; j < INPUT_SIZE; j++)
            network.layers[1].neurons[i].weights[j] = rand() / (double)RAND_MAX;

        network.layers[1].neurons[i].bias = rand() / (double)RAND_MAX;
    }

    return network;
}

void forward_propagation(NeuralNetwork *network, double *inputs) 
{
    for (int i = 0; i < INPUT_SIZE; i++)
        network->layers[0].neurons[i].inputs = &inputs[i];

    for (int i = 0; i < HIDDEN_SIZE; i++) 
    {
        double weighted_sum = 0.0;

        for (int j = 0; j < INPUT_SIZE; j++)
            weighted_sum += network->layers[1].neurons[i].weights[j] * (*(network->layers[0].neurons[j].inputs));

        weighted_sum += network->layers[1].neurons[i].bias;
        network->layers[1].neurons[i].output = sigmoid(weighted_sum);
    }
}

void backward_propagation(NeuralNetwork *network, double *expected_outputs) 
{
    for (int i = 0; i < HIDDEN_SIZE; i++)
        network->layers[1].neurons[i].error = (expected_outputs[i] - network->layers[1].neurons[i].output) * sigmoid_derivative(network->layers[1].neurons[i].output);

    for (int i = 0; i < HIDDEN_SIZE; i++) 
    {
        for (int j = 0; j < INPUT_SIZE; j++)
            network->layers[1].neurons[i].weights[j] += LEARNING_RATE * network->layers[1].neurons[i].error * (*(network->layers[0].neurons[j].inputs));

        network->layers[1].neurons[i].bias += LEARNING_RATE * network->layers[1].neurons[i].error;
    }
}

void train(NeuralNetwork *network, double **training_data, double **expected_outputs, int num_data) 
{
    for (int epoch = 0; epoch < EPOCHS; epoch++) 
    {
        for (int i = 0; i < num_data; i++) 
        {
            forward_propagation(network, training_data[i]);
            backward_propagation(network, expected_outputs[i]);
        }
    }
}

void test(NeuralNetwork *network, double *test_data) 
{
    forward_propagation(network, test_data);

    printf("Hidden Layer Outputs:\n");

    for (int i = 0; i < HIDDEN_SIZE; i++)
        printf("%.4f\n", network->layers[1].neurons[i].output);

    printf("\n");
}

void free_memory(NeuralNetwork *network) 
{
    for (int i = 0; i < HIDDEN_SIZE; i++) 
    {
        free(network->layers[1].neurons[i].inputs);
        free(network->layers[1].neurons[i].weights);
    }

    free(network->layers[0].neurons);
    free(network->layers[1].neurons);
    free(network->layers);
}

int main() 
{
    NeuralNetwork network = initialize_network();

    double *training_data[2];
    training_data[0] = (double[]){0.1, 0.2, 0.3};
    training_data[1] = (double[]){0.4, 0.5, 0.6};

    double *expected_outputs[2];
    expected_outputs[0] = (double[]){0.7, 0.8};
    expected_outputs[1] = (double[]){0.9, 1.0};

    train(&network, training_data, expected_outputs, 2);

    double test_input[INPUT_SIZE] = {0.8, 0.9, 1.0};
    test(&network, test_input);

    free_memory(&network);

    return 0;
}
