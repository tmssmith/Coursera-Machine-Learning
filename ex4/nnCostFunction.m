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




% ======= Part 1 - Cost Function ===========%

% Create structure with all THETA matrices
all_theta(1).val = Theta1;
all_theta(2).val = Theta2;

a(1).val = X;

% Loop through each layer of the Neural Network, computing activation units
% 'a' at each layer
for i = 1:size(all_theta,2)
    a(i).val = [ones(m,1) a(i).val];
    z(i+1).val = a(i).val*all_theta(i).val';
    a(i+1).val = sigmoid(z(i+1).val);
end

h = a(size(all_theta,2)+1).val;

% Map vector 'y' to boolean matrix 'y_bool' where columns 1 through 'num_labels' indicate an output class, rows represent a training example. 
y_bool = 1:num_labels;
y_bool = y==y_bool;

%% Cost Function

% Calculate Cost Function 
J = -1/m*sum(sum(y_bool.*log(h)+((1-y_bool).*log(1-h))));

% Add regularization term to cost function
for k = 1:size(all_theta,2)
    theta = all_theta(k).val;
    J = J + lambda/(2*m)*sum(sum(theta(:,2:end).^2));
end

% Implementation of cost function with for loop
% % Calculate Cost Function for each output class
% for i = 1:num_labels
%     y_bin = y == i;
%     J_temp = 1/m*sum(-y_bin.*log(h(:,i))-((1-y_bin).*log(1-h(:,i))));
%     J = J + J_temp;
% end

%% Back propagation for Theta_grad

delta_3 = a(3).val - y_bool;
delta_2 = delta_3*all_theta(2).val(:, 2:end).*sigmoidGradient(z(2).val);
Theta2_grad = Theta2_grad + delta_3'*a(2).val;
Theta1_grad = Theta1_grad + delta_2'*a(1).val;

% % Implementation of back propagation with For loop
% for t = 1:m
%     % Feed forward to compute 'a'
%     a_1 = [1 X(t,:)]';
%     z_2 = Theta1*a_1;
%     a_2 = sigmoid(z_2);
%     a_2 = [1 ; a_2];
%     z_3 = Theta2*a_2;
%     a_3 = sigmoid(z_3);
%     
%     % Back propagate for delta
%     delta_3 = a_3 - y_bool(t,:)';
%     delta_2 = Theta2'*delta_3.*sigmoidGradient([0 ; z_2]);
%     Theta2_grad = Theta2_grad + delta_3*a_2';
%     Theta1_grad = Theta1_grad + delta_2(2:end)*a_1';
% end

Theta2_grad = 1/m * Theta2_grad;
Theta1_grad = 1/m * Theta1_grad;

% Add regularization term to gradients

Theta2_grad = Theta2_grad + [zeros(size(Theta2,1),1) lambda*Theta2(:,2:end)/m];
Theta1_grad = Theta1_grad + [zeros(size(Theta1,1),1) lambda*Theta1(:,2:end)/m];

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
