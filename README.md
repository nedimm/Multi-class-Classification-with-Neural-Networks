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