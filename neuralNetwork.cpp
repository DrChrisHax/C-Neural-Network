#include "neuralNetwork.h"




int main(int argc, char* argv[]) {

	const double learningRate = 0.1f;

	double hiddenLayer[numHiddenNodes];
	double outputLayer[numOutputs];

	double hiddenLayerBias[numHiddenNodes];
	double outputLayerBias[numOutputs];

	double hiddenWeights[numInputs][numHiddenNodes];
	double outputWeights[numHiddenNodes][numOutputs];

	double training_inputs[numTrainingSets][numInputs] = { {0.0f, 0.0f},
														   {1.0f, 0.0f},
														   {0.0f, 1.0f}, 
														   {1.0f, 1.0f} };

	double training_outputs[numTrainingSets][numOutputs] = { {1.0f},
															 {0.0f},
															 {1.0f},
															 {1.0f} };

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

	int trainingSetOrder[] = { 0, 1, 2, 3 };

	//Training Loop for some number of Epcohs

	for (int epoch = 1; epoch < numOfEpochs + 1; ++epoch) {

		shuffle(trainingSetOrder, numTrainingSets);
		printf("EPOCH: %i\n", epoch);

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
			
			printf("Input p: %g    Input q: %g    Output: %g    Predicted Output: %g\n",
				training_inputs[i][0], training_inputs[i][1],
				outputLayer[0], training_outputs[i][0]);


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

			//fputs("Final Hidden Weights\n[ ", stdout);
			//for (size_t j = 0; j < numHiddenNodes; ++j) {
			//	fputs("[ ", stdout);
			//	for (size_t k = 0; k < numInputs; ++k) {
			//		printf("%f ", hiddenWeights[k][j]);
			//	}
			//}

			//fputs("]\nFinal Hidden Biases\n[ ", stdout);
			//for (size_t j = 0; j < numHiddenNodes; ++j) {
			//	printf("%f ", hiddenLayerBias[j]);
			//}

			//fputs("]\nFinal Output Biases\n[ ", stdout);
			//for (size_t j = 0; j < numOutputs; ++j) {
			//	printf("%f ", outputLayerBias[j]);
			//}

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