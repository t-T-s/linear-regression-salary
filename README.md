# linear-regression-salary
Equation based stastical linear regression model is implemented using python.

The problem is to create a regression model for the variation of salary with the year of experience of employees in company. 

The dataset was divided into two in order to test the model and train the model. First the gradient and intercept for the linear best suited function is calculated. This was calculated as follows.

Gradient = Covariance/Variance 

Intercept= y - Gradient* mean

Then after that the real years of experience is taken and using the regression coefficients we calculated, the predictions for the salary were derived. 
Finally the root mean square error between the test salary and predicted salary values are calculated as an evaluation criteria. 

### Report answers the following questions in order

1. Summarize and compare the characteristics of different Machine Learning techniques/models using a suitable diagram.

2. Draw a flow chart to explain a typical approach to solve a Machine Learning problem.

3. Using an example, explain how the variable importance is calculated in Random Forest classifiers?

4. Implement a simple linear regression model in Python or R using first principles. Use a suitable dataset to test your model. Clearly explain the problem definition
and your approach (i.e., implementation details, accuracy improvements, and performance optimizations etc...). Explain how you can use your model as a simple linear classifier.
