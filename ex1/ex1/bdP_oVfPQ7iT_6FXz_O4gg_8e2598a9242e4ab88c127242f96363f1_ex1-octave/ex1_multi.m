%% Machine Learning Online Class
%  Exercise 1: Linear regression with multiple variables
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear regression exercise. 
%
%  You will need to complete the following functions in this 
%  exericse:
%
%     warmUpExercise.m
%     plotData.m
%     gradientDescent.m
%     computeCost.m
%     gradientDescentMulti.m
%     computeCostMulti.m
%     featureNormalize.m
%     normalEqn.m
%
%  For this part of the exercise, you will need to change some
%  parts of the code below for various experiments (e.g., changing
%  learning rates).
%

%% Initialization

%% ================ Part 1: Feature Normalization ================

%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);


% plot3(data(:,1), data(:,2), data(:,3), 'rx', 'MarkerSize', 10);

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

fprintf('Program paused. Press enter to continue.\n');
%pause;

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

%surf(X, y);

% Add intercept term to X
X = [ones(m, 1) X];



%% ================ Part 2: Gradient Descent ================

% ====================== YOUR CODE HERE ======================
% Instructions: We have provided you with the following starter
%               code that runs gradient descent with a particular
%               learning rate (alpha). 
%
%               Your task is to first make sure that your functions - 
%               computeCost and gradientDescent already work with 
%               this starter code and support multiple variables.
%
%               After that, try running gradient descent with 
%               different values of alpha and see which one gives
%               you the best result.
%
%               Finally, you should complete the code at the end
%               to predict the price of a 1650 sq-ft, 3 br house.
%
% Hint: By using the 'hold on' command, you can plot multiple
%       graphs on the same figure.
%
% Hint: At prediction, make sure you do the same feature normalization.
%

fprintf('Running gradient descent ...\n');

% 从1.162开始，以0.002为间隔作为alpha值
num = (1.2 - 1.0) / 0.02 + 1;
J_examples = zeros(50, round(num));
% 记录不同alpha时得到的theta，因此次所选取的迭代次数最后都逐渐收敛故最终得到的theta都相同
% theta_examples = zeros(3, round(num));    
index = 1;
for i = 1.0 : 0.02 :1.2
    % Choose some alpha value
    alpha = i;
    num_iters = 50;
    theta = zeros(3, 1); % initialize fitting parameters


   % fprintf('\nTesting the cost function ...\n')
   % compute and display initial cost
   % J = computeCostMulti(X, y, theta);
   % fprintf('With theta = [0 ; 0]\nCost computed = %f\n', J);
   % fprintf('=========\n');

    % Init Theta and Run Gradient Descent 

    [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
    
    J_examples(:,index) = J_history;
    %theta_examples(:,index) = theta;
    index = index + 1;
end


% 画出各个alpha取值时得到的代价函数图像
% Plot the convergence graph
figure;
for i = 1 : num
    plot(1:numel(J_history), J_examples(:,i));
    xlabel('Number of iterations');
    ylabel('Cost J');
    hold on;
end

% 画出与pdf中最相似的代价函数图像
figure;
plot(1:numel(J_history), J_examples(:,9));
xlabel('Number of iterations');
ylabel('Cost J');


% alpha = 1.2;
% num_iters = 400;
% 
%  % Init Theta and Run Gradient Descent 
% theta = zeros(3, 1);
% [theta,J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
% 
% % Plot the convergence graph
% figure;
% plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
% xlabel('Number of iterations');
% ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.
% price = 0; % You should change this

% 用原先样本中的sigma和theta进行归一化
XTest = [1650, 3];
XTest = (XTest - mu) ./ sigma;
price = [1, XTest] * theta
% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);

fprintf('Program paused. Press enter to continue.\n');
%pause;

%% ================ Part 3: Normal Equations ================

fprintf('Solving with normal equations...\n');

% ====================== YOUR CODE HERE ======================
% Instructions: The following code computes the closed form 
%               solution for linear regression using the normal
%               equations. You should complete the code in 
%               normalEqn.m
%
%               After doing so, you should complete this code 
%               to predict the price of a 1650 sq-ft, 3 br house.
%

%% Load Data
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');


% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
%price = 0; % You should change this
price_2 = [1, 1650, 3] * theta;



% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price_2);

