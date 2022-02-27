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

para_test = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];   % 测试参数
min_error = Inf;    % 用于记录最小误差
for i = 1 : length(para_test)
    for j = 1 : length(para_test)
        model = svmTrain(X, y, para_test(i), @(x1, x2) gaussianKernel(x1, x2, para_test(j)));   %选取C和sigma用训练集训练
        predictions = svmPredict(model, Xval);   % 用交叉验证集预测
        judge_error = mean(double(predictions ~= yval));    % 用于计算误差,若预测集和真实集则为1，否则为0
        if judge_error < min_error            % 选取出错结果最少的参数C和sigma
            min_error = judge_error;
            C = para_test(i);
            sigma = para_test(j);
        end
    end
end

% =========================================================================

end
