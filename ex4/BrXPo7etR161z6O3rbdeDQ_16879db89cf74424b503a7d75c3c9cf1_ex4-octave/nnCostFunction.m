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

% Training set (X,Y)，数据预处理：将Y转换为目标向量形式
Y = zeros(m, size(Theta2, 1))
for i = 1 : size(Theta2, 1)
    Y(find(y == i), i) = 1; 
end

% Set 各层的输入、输出：a、z
a1 = [ones(m, 1) X];             % input layer 5000x401
z2 = a1 * Theta1';                % hidden layer(in) 5000x25 where Theata 25x401
a2 = [ones(m, 1) sigmoid(z2)];   % hidden layer(out) 5000x26
z3 = a2 * Theta2';                % output layer(in) 5000x10 where Theta2 10x26
a3 = sigmoid(z3)                  % output result
h = a3                            % h_θ(x)


J = 1 / m * sum(sum(-Y .* log(h) - (1 - Y) .* log(1 - h)));

regTerm = lambda / (2 * m) * ...
            (sum(sum(Theta1(:, 2:size(Theta1, 2)).^2)) + ...
             sum(sum(Theta2(:, 2:size(Theta2, 2)).^2)));
J = J + regTerm;

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

delta3 = a3 - Y;                        % 5000x10
delta2 = delta3 * Theta2;               % 5000x26
delta2 = delta2(:, 2:end);              % 5000x25
delta2 = delta2 .* sigmoidGradient(z2); % 5000x25

% Delta表示误差矩阵
Delta2 = zeros(size(Theta2));           % 10x26
Delta1 = zeros(size(Theta1));           % 25x401
Delta2 = Delta2 + delta3' * a2;         % 10x26     a2:5000x26
Delta1 = Delta1 + delta2' * a1;         % 25x401    a1:5000x401

Theta2_grad = 1 / m * Delta2;
Theta1_grad = 1 / m * Delta1;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
Theta1_grad = 1 / m * Delta1 + lambda / m * Theta1;
Theta2_grad = 1 / m * Delta2 + lambda / m * Theta2;
Theta1_grad(:,1) = 1 / m * Delta1(:,1);
Theta2_grad(:,1) = 1 / m * Delta2(:,1);


% -------------------------------------------------------------

% =========================================================================
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
