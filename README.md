Neural Network for Regression
=============================

This repository shows how to use a neural network for predicting real values.

The problem used is given weight in lbs and height in inches, calculate the body mass index (BMI).

This is still work in progress as I am figuring out how to use a neural network for regression.

Following changes were made:

1. Change the cost function to cost function for regression.
2. Don't use sigmoid activation function, but use the activation function in linear regression i.e. h(Theta) = Theta' * X;
3. Remove the derivative term for the sigmoid cost function in the back-propagation part.

Sample output is:

```cmd
Height : 68 Weight : 160, predicted BMI : 24.437453, actual BMI : 24.325260
Height : 72 Weight : 200, predicted BMI : 27.824245, actual BMI : 27.121914
Height : 60 Weight : 100, predicted BMI : 20.818450, actual BMI : 19.527778
Height : 68 Weight : 165, predicted BMI : 25.226098, actual BMI : 25.085424
Height : 69 Weight : 226, predicted BMI : 34.116974, actual BMI : 33.370720
Height : 71 Weight : 230, predicted BMI : 33.286706, actual BMI : 32.074985
Height : 60 Weight : 105, predicted BMI : 21.607095, actual BMI : 20.504167
```