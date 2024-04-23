#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

//Defs for NN structure
#define numInputs 2
#define numHiddenNodes 2
#define numOutputs 1
#define numTrainingSets 4

#define numOfEpochs 10000


double init_weights();
double sigmoid(double x);
double dSigmoid(double x);
void shuffle(int* array, size_t n);


#endif

