function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
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

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Fast forward

% transfer y to Y
% Ex: 10 -> [0 0 0 0 0 0 0 0 0 1]
% Ex: 2  -> [0 1 0 0 0 0 0 0 0 0]
I = eye(num_labels);
Y = zeros(m, num_labels);
for i=1:m
  Y(i, :)= I(y(i), :);
end

% first layer (input) -> second layer
X_1 = [ones(m,1), X];
Z_2 = X_1 * Theta1';
A_2 = sigmoid(Z_2);

% second layer -> third layer (output)
A_2 = [ones(size(A_2, 1), 1), A_2];
Z_3 = A_2 * Theta2';
A_3 = sigmoid(Z_3);

% penalty for regulurization
% dont count the bias (Theta_0)
penalty = sum(sum(Theta1(:, 2:end).^2)) + sum(sum(Theta2(:, 2:end).^2))

% cost function with regularization
J = sum(sum((-Y).*log(A_3) - (1-Y).*log(1-A_3), 2))/m + lambda / (2*m) * penalty

% -------------------------------------------------------------

% Backward
% step 2, output layer
sigma_3 = A_3.-Y

% step 3, hidden layer 2 (second layer)
sigma_2 = sigma_3 * Theta2 .* sigmoidGradient([ones(size(Z_2, 1), 1), Z_2])
sigma_2 = sigma_2(:, 2:end);

% step4, Accumulate the gradient
delta_1 = (sigma_2' * X_1);
delta_2 = (sigma_3' * A_2);

% penalty for regularization
% index 1 dont penalty
penalty1 = (lambda / m) * [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
penalty2 = (lambda / m) * [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];


% calculate regularized gradient
Theta1_grad = delta_1./m + penalty1;
Theta2_grad = delta_2./m + penalty2;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
