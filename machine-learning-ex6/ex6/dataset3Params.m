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
possible_C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
possible_sigma = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
predict_error_result = zeros(size(possible_C, 2), size(possible_sigma, 2));

col_index = 1
row_index = 1
for c_i = possible_C
    for sigma_i = possible_sigma
        model= svmTrain(X, y, c_i, @(x1, x2) gaussianKernel(x1, x2, sigma_i));
        predictions = svmPredict(model, Xval);
        predict_error = mean(double(predictions ~= yval))
        predict_error_result(row_index, col_index) = predict_error
        col_index += 1
    end
    row_index += 1
    col_index = 1
end
[minval, col] = min(min(predict_error_result,[],1))
[minval, row] = min(min(predict_error_result,[],2))

C = possible_C(row)
sigma = possible_sigma(col)
% =========================================================================

end
