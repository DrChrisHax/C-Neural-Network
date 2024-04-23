#include "neuralNetwork.h"




int main(int argc, char* argv[]) {

	const double learningRate = 0.1f;

	double totalGrade = 0.0f;

	double hiddenLayer[numHiddenNodes];
	double outputLayer[numOutputs];

	double hiddenLayerBias[numHiddenNodes];
	double outputLayerBias[numOutputs];

	double hiddenWeights[numInputs][numHiddenNodes];
	double outputWeights[numHiddenNodes][numOutputs];

	double training_inputs[numTrainingSets][numInputs] = 
	{
		//A, B, Cin
		{0.0f, 0.0f, 0.0f},
		{0.0f, 0.0f, 1.0f},
		{0.0f, 1.0f, 0.0f},
		{1.0f, 0.0f, 0.0f},
		{0.0f, 1.0f, 1.0f},
		{1.0f, 1.0f, 0.0f},
		{1.0f, 0.0f, 1.0f},
		{1.0f, 1.0f, 1.0f}
	};

	double training_outputs[numTrainingSets][numOutputs] = 
	{ 
		//Sum, Cout
		{0.0f, 0.0f},
		{1.0f, 0.0f},
		{1.0f, 0.0f},
		{1.0f, 0.0f},
		{0.0f, 1.0f},
		{0.0f, 1.0f},
		{0.0f, 1.0f},
		{1.0f, 1.0f}
	};

	int trainingSetOrder[] = { 0, 1, 2, 3, 4, 5, 6, 7 };

	for (size_t i = 0; i < numInputs; ++i) {
		for (size_t j = 0; j < numHiddenNodes; ++j) {
			hiddenWeights[i][j] = init_weights();
		}
	}

	for (size_t i = 0; i < numHiddenNodes; ++i) {
		hiddenLayerBias[i] = init_weights();
	}

	for (size_t i = 0; i < numHiddenNodes; ++i) {
		for (size_t j = 0; j < numOutputs; ++j) {
			outputWeights[i][j] = init_weights();
		}
	}

	for (size_t i = 0; i < numOutputs; ++i) {
		outputLayerBias[i] = init_weights();
	}

	//Training Loop for some number of Epcohs
	for (int epoch = 1; epoch < numOfEpochs + 1; ++epoch) {

		shuffle(trainingSetOrder, numTrainingSets);


		if(epoch % 100 == 0) { printf("\nEPOCH: %i\n", epoch); }
		double totalGrade = 0.0f;

		for (size_t x = 0; x < numTrainingSets; ++x) {
			int i = trainingSetOrder[x];

			//Forward pass
			//Compute hidden layer activation 
			for (size_t j = 0; j < numHiddenNodes; j++) {
				double activation = hiddenLayerBias[j];

				for (size_t k = 0; k < numInputs; k++) {
					activation += training_inputs[i][k] * hiddenWeights[k][j];
				}

				hiddenLayer[j] = sigmoid(activation);
			}	
			//Compute output layer activation 
			for (size_t j = 0; j < numOutputs; j++) {
				double activation = outputLayerBias[j];
				for (size_t k = 0; k < numHiddenNodes; k++) {
					activation += hiddenLayer[k] * outputWeights[k][j];
				}

				outputLayer[j] = sigmoid(activation);
			}

			//Output weights
			if (epoch % 100 == 0) {
				for (int j = 0; j < numInputs; ++j) {
					printf("Input %i: %g \n", j, training_inputs[i][j]);
				}
				for (int j = 0; j < numOutputs; ++j) {
					printf("Expected Output %i: %g \n", j, training_outputs[i][j]);
				}
				for (int j = 0; j < numOutputs; ++j) {
					printf("Actual output %i: %g \n", j, outputLayer[j]);
				}

				//Grade Model
				double grade = 1 - calculateMSE(outputLayer, training_outputs[i], numOutputs);
				totalGrade += grade;
				printf("Accuracy(MSE): %g\n", grade);

				
			}

			//Back Propagation
			//Compute change in output weights
			double deltaOutput[numOutputs];

			for (size_t j = 0; j < numOutputs; ++j) {
				double error = (training_outputs[i][j] - outputLayer[j]);
				deltaOutput[j] = error * dSigmoid(outputLayer[j]);
			}

			//Compute change in hidden weights
			double deltaHidden[numHiddenNodes];
			for (size_t j = 0; j < numHiddenNodes; ++j) {
				double error = 0.0f;
				for (size_t k = 0; k < numOutputs; ++k) {
					error += deltaOutput[k] * outputWeights[j][k];
				}
				deltaHidden[j] = error * dSigmoid(hiddenLayer[j]);
			}

			//Apply changes in output weights
			for (size_t j = 0; j < numOutputs; ++j) {
				outputLayerBias[j] += deltaOutput[j] * learningRate;
				for(size_t k = 0; k < numHiddenNodes; ++k) {
					outputWeights[k][j] += hiddenLayer[k] * deltaOutput[j] * learningRate;
				}
			}

			//Apply changes in hidden weights
			for (size_t j = 0; j < numHiddenNodes; ++j) {
				hiddenLayerBias[j] += deltaHidden[j] * learningRate;
				for (size_t k = 0; k < numInputs; ++k) {
					hiddenWeights[k][j] += training_inputs[i][k] * deltaHidden[j] * learningRate;
				}
			}
		}
		if (epoch % 100 == 0) {
			totalGrade = totalGrade / numTrainingSets;
			printf("Total accuracy for Epoch: %g \n", totalGrade);
		}
		

	}
}




double init_weights() { return ((double) rand()) / ((double) RAND_MAX); }
double sigmoid(double x) { return 1 / (1 + exp(-x)); }
double dSigmoid(double x) { return x * (1 - x); }
void shuffle(int* array, size_t n) {
	if (n > 1) {
		for (size_t i = 0; i < n - 1; ++i) {
			size_t j = i + rand() / ((RAND_MAX / (n - i)) + 1);
			int temp = array[j];
			array[j] = array[i];
			array[i] = temp;
		}

	}
}
double calculateAccuracy(const double actual[], const double predicted[], size_t n) {
	int correct = 0;
	for (size_t i = 0; i < n; ++i) {
		if (actual[i] >= 0.5f) { ++correct; }
	}
	return (double)correct / n;
}

double calculateMSE(const double actual[], const double predicted[], size_t n) {
	double sumSquareError = 0.0f;

	for (size_t i = 0; i < n; ++i) {
		double squaredError = pow(actual[i] - predicted[i], 2);
		sumSquareError += squaredError;
	}
	return sumSquareError / n;
}

double calculateRMSE(const double actual[], const double predicted[], size_t n) {
	double sum_squared_diff = 0.0f;
	for (size_t i = 0; i < n; ++i) {
		double diff = actual[i] - predicted[i];
		sum_squared_diff += diff * diff;
	}
	return sqrt(sum_squared_diff / n);
}