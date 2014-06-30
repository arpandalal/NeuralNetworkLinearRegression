function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a three layer
%neural network which performs classification, where there's one hidden layer
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% m represents the size of training dataset
m = size(X, 1);
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

p1 = sigmoid([ones(m,1) X] * Theta1');
p2 = sigmoid([ones(m, 1) p1] * Theta2');

% ---- The cost calculation for the training set
J =sum((p2 - y) .^ 2) / (2 * m);

% ------------------------------------------------------------
% Applying regularization on the calculated cost
reg1 = sum(sum(Theta1(:, 2:size(Theta1, 2)) .^2));
reg2 = sum(sum(Theta2(:, 2:size(Theta2, 2)) .^2));

J = J + lambda / (2 * m) * (reg1 + reg2);


% -------------------------------------------------------------
% Back propagation algorithm


for i = 1:m
  
  % --- Step 1 --- Perform feedforward pass
  a1 = [ 1 X(i, :)];
  z2 = a1 * Theta1';
  a2 = [1 sigmoid(z2)];
  z3 = a2 * Theta2';
  a3 = sigmoid(z3);
  
  % --- Step 2 --- Calculate the error in the output layer
  delta3 = a3 - y(i);
  
  % --- Step 3 --- Calculate error for the hidden layer
  delta2 = (delta3 * Theta2);  %.* [1 sigmoidGradient(z2)];

  % --- Step 4 --- Accumulate the gradients
 
  delta2 = delta2(2:end);
  Theta1_grad = Theta1_grad + delta2' * a1;
  Theta2_grad = Theta2_grad + delta3' * a2;
end

% --- step 5 -------------
% Divide the gradients by m (total number of samples)

  Theta1_grad = Theta1_grad / m;
  reg1 = Theta1(:, 2:size(Theta1, 2)) .* (lambda / m);
  reg1 = [zeros(size(Theta1, 1), 1) reg1];
  Theta1_grad = Theta1_grad + reg1;

  Theta2_grad = Theta2_grad / m; 
  reg2 = Theta2(:, 2:size(Theta2, 2)) .* (lambda / m);
  reg2 = [zeros(size(Theta2, 1), 1) reg2];
  Theta2_grad = Theta2_grad + reg2;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
