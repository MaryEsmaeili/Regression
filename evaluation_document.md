## Custom vs. Sklearn Regression Models
### Observations
- Custom Linear Regression MSE: 1.9437 vs Sklearn: 0.1549
- Custom Logistic Regression Accuracy: 0.8047 vs Sklearn: 0.8062
---
### Why Custom Linear Regression MSE is Higher
1. Feature Scaling: Sklearn might have carried out the feature preprocessing for example centering on its own. Make sure both algorithms have the same feature scaling (e.g., `StandardScaler`) before comparison.
2. Intercept Handling: The custom model adds the intercept explicitly while `sklearn` does it internally. The intercept of `sklearn` can be made one of the two - either it is free, or it is fixed and you let it handle the rest.
3. Regularization: While the custom model executes regularization (`lambda_`), on the other hand, `sklearn`'s default `LinearRegression` does not use such regularization. To get more legitimate result validate by `Ridge`.
4. Gradient Descent: Sklearn uses advanced solvers, but the custom method applies gradient descent. Hyperparameters tuning (e.g. learning rate, iterations) can help get better convergence and smaller errors.
---
#### Why Logistic Regression Models are Similar
1. Logistic Regression Simplicity: In fact, the simpler the model, the more resistant it is to scaling and intercept handling factors, which provides an explanation for their identical performance.
2. Regularization: The regularization term played a small part in linear regression success for this  case.
3. Gradient Descent: The custom gradient descent probably converged very well, thus the outputs which were almost the same with `sklearn`.
