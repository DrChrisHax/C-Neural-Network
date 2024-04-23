#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

//Defs for NN structure
#define numInputs 3
#define numHiddenNodes 3
#define numOutputs 2
#define numTrainingSets 8 //2^numInputs

#define numOfEpochs 10000


double init_weights();
double sigmoid(double x);
double dSigmoid(double x);
void shuffle(int* array, size_t n);

//Functions for grading the model
double calculateAccuracy(const double actual[], const double predicted[], size_t n);
double calculateMSE(const double actual[], const double predicted[], size_t n);
double calculateRMSE(const double actual[], const double predicted[], size_t n);




#endif

