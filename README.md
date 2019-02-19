# Multi-class Classification with Neural Networks
Implementation one-vs-all logistic regression with neural networks to recognize hand-written digits.
In this project we will implement one-vs-all logistic regression with neural networks to recognize hand-written digits. 
The project is an exercise from the ["Machine Learning" course from Andrew Ng](https://www.coursera.org/learn/machine-learning/) .

The implementation was done using GNU Octave. We will start with the scripts `ex3.m` and `ex3_nn.m`.
These scripts set up the dataset for the problems and make calls to functions that we will write in separate `*.m` files.

## Multi-class Classification

For this project we will use logistic regression and neural networks to recognize handwritten digits (from 0 to 9). Automated handwritten digit recognition is widely used today - from recognizing zip codes (postal codes) on mail envelopes to recognizing amounts written on bank checks. In this part of the project, we will use extend the previous implementation of logistic regression and apply it to one-vs-all classification.

### Dataset

Data set is in `ex3data1.mat` that contains 5000 training examples of handwritten digits. The `.mat` format means that the data has been saved in a native Octave/Matlab matrix format, instead of a text (ASCII) formal like a csv-file. These matrices can be read directly into your program by using the load command. After loading, matrices of the correct
dimensions and values will appear in your program’s memory. The matrix will already be named, so you do not need to assign names to them. 
There are 5000 training examples in `ex3data1.mat`, where each training example is a 20 pixel by 20 pixel grayscale image of the digit. Each pixel is represented by a floating point number indicating the grayscale intensity at
that location. The 20 by 20 grid of pixels is "unrolled" into a 400-dimensional vector. Each of these training examples becomes a single row in our data matrix X. This gives us a 5000 by 400 matrix X where every row is a training
example for a handwritten digit image.

![X](https://i.imgur.com/jPCwZRx.png)

The second part of the training set is a 5000-dimensional vector y that contains labels for the training set. To make things more compatible with Octave/Matlab indexing, where there is no zero index. The digit zero is mapped to the value ten. Therefore, a \0" digit is labeled as \10", while the digits \1" to \9" are labeled as \1" to \9" in their natural order.

## Visualizing the data

We will begin by visualizing a subset of the training set. In Part 1 of `ex3.m`, the code randomly selects 100 rows from X and passes those rows to the displayData function. This function maps each row to a 20 pixel by 20 pixel grayscale image and displays the images together. 
```matlab
%% =========== Part 1: Loading and Visualizing Data =============
fprintf('Loading and Visualizing Data ...\n')
load('ex3data1.mat'); % training data stored in arrays X, y
m = size(X, 1);
% Randomly select 100 data points to display
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);
displayData(sel);
```

The function for displaying the data is stored in `displayData.m`. 
After you run this step, you should see an image like shown in Figure 1.

![input_data](https://i.imgur.com/1qspUIr.png)
***Figure 1: Examples from the dataset***

### Vectorizing Logistic Regression

We will be using multiple one-vs-all logistic regression models to build a multi-class classifier. Since there are 10 classes, we will need to train 10 separate logistic regression classifiers. To make this training efficient, it is
important to ensure that our code is well vectorized. In this section, we will implement a vectorized version of logistic regression that does not employ any `for` loops.

#### Vectorizing the cost function
We will begin by writing a vectorized version of the cost function. Recall that in (unregularized) logistic regression, the cost function is:

![unvectorized cost function](https://i.imgur.com/5uZepmE.png)

To compute each element in the summation, we have to compute $$h_θ(x(i))$$ for every example `i`, where 
$$h_θ(x(i)) = g(θ^T x(i))$$ and $$g(z)=\frac{1}{1+e^{-z}}$$ is the sigmoid function. It turns out that we can compute this quickly for all our examples by using matrix multiplication. Let us define X and θ as

![](https://i.imgur.com/qd0IgpB.png)

Then, by computing the matrix product Xθ, we have

![](https://i.imgur.com/3uoQVdg.png)

In the last equality, we used the fact that $$a^Tb = b^Ta$$ if `a` and `b` are vectors.
This allows us to compute the products $$θ^Tx^{(i)}$$ for all our examples `i` in one line of code.

#### Vectorizing the gradient
Recall that the gradient of the (unregularized) logistic regression cost is a vector where the $$j^{th}$$ element is defined as

![gradient](https://i.imgur.com/Ut9Q1fC.png)

To vectorize this operation over the dataset, we start by writing out all the partial derivatives explicitly for all $$θ_j$$,

![](https://i.imgur.com/IYHN1rE.png)

where

![](https://i.imgur.com/5OEnkfk.png)

Note that $$x^{(i)}$$ is a vector, while $$(h_θ(x^{(i)})-y^{(i)})$$ is a scalar (single number).
To understand the last step of the derivation, let $$β_{i}=(h_{θ}(x^{(i)})-y^{(i)})$$ and
observe that:

![](https://i.imgur.com/55xa2xV.png)

The expression above allows us to compute all the partial derivatives without any loops.

#### Vectorizing regularized logistic regression
After we have implemented vectorization for logistic regression, we will now add regularization to the cost function. Recall that for regularized logistic regression, the cost function is defined as

![regularized logistic regression](https://i.imgur.com/UfPtUqf.png)

Note that you should not be regularizing $$θ_0$$ which is used for the bias term.
Correspondingly, the partial derivative of regularized logistic regression cost for $$θ_j$$ is defined as

![](https://i.imgur.com/zcRIeA9.png)

Regularized cost function is implemented in the file `lrCostFunction.m`
```matlab
function [J, grad] = lrCostFunction(theta, X, y, lambda)
    m = length(y); % number of training examples
    J = ( (1 / m) * sum(-y'*log(sigmoid(X*theta)) - (1-y)'*log( 1 - sigmoid(X*theta))) ) + (lambda/(2*m))*sum(theta(2:length(theta)).*theta(2:length(theta))) ;
    
    grad = (1 / m) * sum( X .* repmat((sigmoid(X*theta) - y), 1, size(X,2)) );
    grad(:,2:length(grad)) = grad(:,2:length(grad)) + (lambda/m)*theta(2:length(theta))';
    grad = grad(:);
end
```

### One-vs-all Classification

In this part of the exercise, we will implement one-vs-all classification by training multiple regularized logistic regression classifiers, one for each of the K classes in our dataset. In the handwritten digits dataset,
K = 10, but our code should work for any value of K.
The code in `oneVsAll.m` trains one classifier for each class.

```matlab
function [all_theta] = oneVsAll(X, y, num_labels, lambda)
    m = size(X, 1);
    n = size(X, 2);
    all_theta = zeros(num_labels, n + 1);
    X = [ones(m, 1) X];
    initial_theta = zeros(n + 1, 1);
    options = optimset('GradObj', 'on', 'MaxIter', 50);
    for c = 1:num_labels
        all_theta(c,:) = fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), initial_theta, options);
end
```
Note that the `y` argument to this function is a vector of labels from 1 to 10, where we have mapped the digit "0" to the label 10 (to avoid confusions with indexing).

#### One-vs-all Prediction

After training our one-vs-all classifier, we can now use it to predict the digit contained in a given image. For each input, we should compute the "probability" that it belongs to each class using the trained logistic regression
classifiers. Our one-vs-all prediction function will pick the class for which the corresponding logistic regression classifier outputs the highest probability and return the class label (1, 2,..., or K) as the prediction for the input example.
The code in `predictOneVsAll.m` uses the one-vs-all classifier to make predictions.
```matlab
function p = predictOneVsAll(all_theta, X)
    m = size(X, 1);
    num_labels = size(all_theta, 1);
    p = zeros(size(X, 1), 1);
    X = [ones(m, 1) X];
    for i = 1:m
        RX = repmat(X(i,:),num_labels,1);
        RX = RX .* all_theta;
        SX = sum(RX,2);
        [val, index] = max(SX);
        p(i) = index;
    end
end
```
In `ex3.m` we will call `predictOneVsAll` function using the learned value of Θ. You should see that the training set accuracy is about 94.9% (i.e., it classifies 94.9% of the examples in the training set correctly).

## Neural Networks

In this part we will implement a neural network to recognize handwritten digits using the same training set as before. The neural network will be able to represent complex models that form non-linear hypotheses. We will be using parameters from a neural network that have been already trained. Our goal is to implement the feedforward
propagation algorithm to use our weights for prediction. In next project, we will write the backpropagation algorithm for learning the neural network parameters.
We start with the provided script `ex3_nn.m`.

### Model representation

Our neural network is shown in Figure 2. It has 3 layers - an input layer, a hidden layer and an output layer. Recall that our inputs are pixel values of digit images. Since the images are of size 20×20, this gives us 400 input layer units (excluding the extra bias unit which always outputs +1). As before, the training data will be loaded into the variables X and y.
You have been provided with a set of network parameters $$(Θ^{(1)},Θ^{(2)})$$ already trained. These are stored in `ex3weights.mat` and will be loaded by `ex3_nn.m` into Theta1 and Theta2. The parameters have dimensions that are sized for a neural network with 25 units in the second layer and 10 output units (corresponding to the 10 digit classes).

![](https://i.imgur.com/cyIj1JY.png)
***Figure 2: Neural network model***

### Feedforward Propagation and Prediction

Now we will implement feedforward propagation for the neural network. We will need to complete the code in `predict.m` to return the neural network’s prediction.
You shall implement the feedforward computation that computes $$h_θ(x^{(i)})$$ for every example `i` and returns the associated predictions. Similar to the one-vs-all classification strategy, the prediction from the neural network will
be the label that has the largest output $$(h_θ(x))_k$$.

```matlab
function p = predict(Theta1, Theta2, X)
    m = size(X, 1);
    num_labels = size(Theta2, 1);
    p = zeros(size(X, 1), 1);
    a1 = [ones(m, 1) X];
    z2 = a1 * Theta1';
    a2 = sigmoid(z2);
    a2 = [ones(size(a2,1), 1) a2];
    z3 = a2 * Theta2';
    a3 = sigmoid(z3);
    [val, index] = max(a3,[],2);
    p = index;
end
```

Now we can execute `ex3_nn.m` which will call our predict function using the loaded set of parameters for Theta1 and Theta2. You should see that the accuracy is about 97.5%. After that, an interactive sequence will launch displaying images from the training set one at a time, while the console prints out the predicted label for the displayed image. 

![](https://i.imgur.com/uBNPJ6V.png)

![](https://i.imgur.com/Ft8oVuv.png)

![](https://i.imgur.com/kvAhul4.png)

![](https://i.imgur.com/xk0G8Rw.png)

![](https://i.imgur.com/43G2uw1.png)
