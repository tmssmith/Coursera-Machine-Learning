function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_set = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_set = [0.01 0.03 0.1 0.3 1 3 10 30];
error = zeros(8,8);

% Iterate through each C value
for i = 1:8
    C = C_set(i);
    % Iterate through each sigma value
    for j = 1:8
        sigma = sigma_set(j);
        % Train model for given theta and sigma using training data set
        model = svmTrain(X,y,C,@(x1, x2)gaussianKernel(x1, x2, sigma));
        % Predict outputs of validation data set
        predictions = svmPredict(model,Xval);
        % Compute errors of predictions vs actual outputs
        error(i,j) = mean(predictions ~= yval);
    end
end

% Identify indices of minimum error value
[~,I] = min(error,[],"all","linear");
[C_ind, sigma_ind] = ind2sub(size(error),I);
C = C_set(C_ind);
sigma = sigma_set(sigma_ind);
% =========================================================================

end
