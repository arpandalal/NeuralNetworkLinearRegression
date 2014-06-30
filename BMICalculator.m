% Main entry script to train the neural network
clear ; close all; clc

trainingDataFile = "bmidata_normalized.csv"
input = load(trainingDataFile, "");
X = input(:, 1:2);
y = input(:, 3);

fprintf('\nData Loaded');

% Determine the type of neural network. Input layer with 2 nodes, one hidden layer with 4 nodes and 4 output nodes.
input_layer_size = 2;
hidden_layer_size = 500;
num_labels = 1;
m = size(X, 1);

% Randomly generate initial weights for symmetry breaking
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


fprintf('\nTraining Neural Network... \n')

% Train the neural network for specified number of iterations.
options = optimset('MaxIter', 2000);

% Set regularization
lambda = 0.01;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
% Theta1 is the model parameters for the hidden layer and Theta2 for the output layer
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Predict the values for entire input and determine the training set accuracy.
% Save the hidden layer parameters in a file named Theta1.txt
% Save the output layer parameters in a file named Theta2.txt
% Save the predicted results in a file named PredictedResults.txt

pred = predict(Theta1, Theta2, X);
save -ascii "Theta1.txt" Theta1;
save -ascii "Theta2.txt" Theta2;

actualY = y * (51 - 8) + 23.96;
predictedY = pred * (51 - 8) + 23.96;

save -ascii  "PredictedResults.txt" predictedY;
save -ascii "ActualResults.txt" actualY;

percentError = abs(predictedY - actualY) ./ actualY * 100;
save -ascii "PercentageError.txt" percentError;

fprintf('\nTraining Set Accuracy: %f\n', mean(percentError <= 5) * 100);


testData = [68 160 ; 72 200; 60 100; 68 165; 69 226; 71 230; 60 105];
normTestData = [(testData(:,1) - 67.845)/(78 - 58) (testData(:,2) - 156.255)/(250 - 70)];
pred = predict(Theta1, Theta2, normTestData);
finalValue = (pred) * (51 - 8) + 23.96;
fprintf('Height : %d Weight : %d, predicted BMI : %f, actual BMI : %f\n', testData(1, 1), testData(1,2), finalValue(1), testData(1,2)/ (testData(1,1) ^ 2) * 703);
fprintf('Height : %d Weight : %d, predicted BMI : %f, actual BMI : %f\n', testData(2, 1), testData(2,2), finalValue(2), testData(2,2)/ (testData(2,1) ^ 2) * 703);
fprintf('Height : %d Weight : %d, predicted BMI : %f, actual BMI : %f\n', testData(3, 1), testData(3,2), finalValue(3), testData(3,2)/ (testData(3,1) ^ 2) * 703);
fprintf('Height : %d Weight : %d, predicted BMI : %f, actual BMI : %f\n', testData(4, 1), testData(4,2), finalValue(4), testData(4,2)/ (testData(4,1) ^ 2) * 703);
fprintf('Height : %d Weight : %d, predicted BMI : %f, actual BMI : %f\n', testData(5, 1), testData(5,2), finalValue(5), testData(5,2)/ (testData(5,1) ^ 2) * 703);
fprintf('Height : %d Weight : %d, predicted BMI : %f, actual BMI : %f\n', testData(6, 1), testData(6,2), finalValue(6), testData(6,2)/ (testData(6,1) ^ 2) * 703);
fprintf('Height : %d Weight : %d, predicted BMI : %f, actual BMI : %f\n', testData(7, 1), testData(7,2), finalValue(7), testData(7,2)/ (testData(7,1) ^ 2) * 703);

%testData2 = [72, 200];
%pred2 = predict(Theta1, Theta2, testData2);
%finalValue2 = (pred2) * (51 - 8) + 23.96
%fprintf('Height : %d Weight : %d, predicted BMI : %f, actual BMI : %f\n', testData2(1), testData2(2), finalValue2, testData2(2)/ (testData2(1) ^ 2) * 703);

%testData3 = [60, 100];
%pred3 = predict(Theta1, Theta2, testData3);
%finalValue3 = (pred3) * (51 - 8) + 23.96
%fprintf('Height : %d Weight : %d, predicted BMI : %f, actual BMI : %f\n', testData3(1), testData3(2), finalValue3, testData3(2)/ (testData3(1) ^ 2) * 703);
