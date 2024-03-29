# JML
A simple, easy-to-use modular machine learning library for Java. This library uses the [Java-Linear-Algebra](https://github.com/jacobdwatters/Java-Linear-Algebra)
library to satisfy any linear algebra needs. It is highly recommended that you also have the .jar file
for this library as well.

### Install
1) Ensure you have [Java SE 16](https://www.oracle.com/java/technologies/javase/jdk17-archive-downloads.html) or later installed.
2) Download the Jar file for this library [here](https://www.worldofjacobwatters.com/jml-download)
   1) Also download the Jar file for the Java-Linear-Algebra library [here](https://www.worldofjacobwatters.com/download)
3) Add the Jar(s) to your project in your IDE of choice. 
4) You are ready to being developing machine learning models in Java!

### Documentation and Use
1) The Javadoc for the JML library can be found [here](https://jacobdwatters.github.io/JML/index.html). 
2) Examples of using the library can be found on my personal website [here](https://www.worldofjacobwatters.com/java-machine-learning-use).

## Features
>This library allows the user to do the following and more:
>- <b>Fit and train a variety of models</b>
>  - <b>Regression models</b>
>    - Linear regression
>    - Multiple linear regression
>    - Polynomial regression
>  - <b>Classification models</b>
>    - K-Nearest neighbors
>    - Logistic Regression
>    - Gaussian Naive Bayes
>    - perceptrons
>  - <b>Neural networks</b>
>    - Layers
>      - Linear
>      - Dense
>      - Dropout
>    - Activation functions
>      - linear
>      - ReLU (rectified linear unit)
>      - Sigmoid
>      - tanh (hyperbolic tangent)
>    - Loss Functions
>      - Mean squared error
>      - Binary cross-entropy
>      - Multi-class cross-entropy
>    - Weight and Bias initializers
>      - Constant values
>      - Random normal/uniform values 
>        - Specify mean, standard deviation, min, and max.
>      - Glorot normal/uniform values
>      - He normal/uniform values
>      - Random orthongonal matrices
>        - matrix must be square.
>- <b>Optimizers</b>
>     - Gradient descent
>     - Momentum
>     - Adam
>- <b>Add learning rate schedulers to optimizer</b>
>- <b>Use trained models to make predictions on new data</b>
>- <b>Save and reuse trained models</b>
>     - Save a trained model to a file for later use
>     - Load a saved model and make predictions with it
>- <b>Manipulate Data</b>
>    - Read and write data from and to a file
>    - Encode text labels
>    - Split data into training and testing sets
>    - Normalize data
>    - Standardize data
>- <b>Compute statistics of data</b>
>     - Mean
>     - Median
>     - Mode
>     - Standard Deviation
>     - Variance

Please note this library is still under development. Optimizations and new features are planned.

### A Word of Caution
This library was primary developed for educational purposes. While efforts are currently being made to make this library fast and reliable,
other Java libraries with similar features may be better suited for large scale practical applications.
