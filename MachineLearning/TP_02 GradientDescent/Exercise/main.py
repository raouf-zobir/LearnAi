import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LinearRegression

def using_LinearRegression():
    df = pd.read_csv("test_scores.csv")
    r = LinearRegression()
    r.fit(df[['math']],df.cs)
    return r.coef_, r.intercept_

def using_Gradient_Descent(x, y):
    m_current = 0
    b_current = 0
    iterations = 10000
    n = len(x)
    learning_rate = 0.0002
    cost_previous = 0


    for i in range(iterations):
        y_predicted = m_current * x + b_current
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)])
        
        m_derivative = -(2/n)* sum(x*(y-y_predicted))
        b_derivative = -(2/n)* sum((y-y_predicted))

        m_current = m_current - (learning_rate * m_derivative)
        b_current = b_current - (learning_rate * b_derivative)

        if math.isclose(cost, cost_previous, rel_tol=1e-20):
            break
        cost_previous = cost
        print ("m {}, b {}, cost {}, iteration {}".format(m_current,b_current,cost, i))
            
    return m_current, b_current

df = pd.read_csv('test_scores.csv')
x = np.array(df.math)
y = np.array(df.cs)

m, b = using_Gradient_Descent(x, y)
print("Using Gradient descent  : m = [{}] / b = [{}]".format(m, b))

m, b = using_LinearRegression()
print("Using Linear Regression : m = {} / b = {}".format(m, b))
