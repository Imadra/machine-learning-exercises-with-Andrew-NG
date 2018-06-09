function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
coef = 1 / m;
h = sigmoid(X*theta);
first = -y.*log(h);
second = -(1-y).*log((1-h));
all = sum(first + second);
n = size(theta);
temp = theta(2:n);
secall = sum(temp.^2);
J = coef * all +(lambda/(2*m))*secall;
grad(1) = coef*sum((h-y)'*X(:,1));
for i=2:n
	grad(i) = coef*sum((h-y)'*X(:,i)) + (lambda/m)*theta(i);
end
% =============================================================

end