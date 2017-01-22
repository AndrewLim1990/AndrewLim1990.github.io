---
title:  "Recommending Movies with Machine Learning"
date:   2017-01-21 15:04:23
categories: [machine-learning]
tags: [machine-learning]
header:
  image: /assets/images/tiger.jpg
  caption: "Photo credit: [**Unsplash**](https://unsplash.com)"
---


In this blog post, I'll be going through a toy example of how to make a recommendation system using a collaborative filtering algorithm.

This blog post has been adapted from a lab exercise from the Stanford Machine Learning course on Coursera found [here](https://www.coursera.org/learn/machine-learning)

All code and data can be found in [this github repository](https://github.com/AndrewLim1990/recommender_post)

## Problem Formulation:

As motivation for this problem, let's pretend that we work for Netflix and want to be able to predict how a user, Kitson, would rate a movie "Two Lovers". One way of doing this is looking at Kitson's past ratings for movies similar to "Two Lovers" and check if he had given them a high rating. If he has, we can guess that he would also give a high rating for "Two Lovers". Additionally, we can use the data from other users to see if those with the same preferences as Kitson had also rated the movie highly. If they have, this is further evidence that Kitson would rate the movie highly. Using these two pieces of information, we can take "Two Lovers" and put it on Kitson's movie feed when he logs in causing him to be happy and want to continue our service!

Let's explore this problem with an example:

## Implementation:


```python
# Loading libraries:
import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Reading in data
y_raw = pd.read_csv("../data/small_movies.csv")
y = y_raw.drop("Movie", axis=1)
y_raw
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Movie</th>
      <th>Matt</th>
      <th>Kitson</th>
      <th>Subi</th>
      <th>Pratheek</th>
      <th>Andrew</th>
      <th>Rodrigo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Love Everywhere</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>0.5</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Two Lovers</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.1</td>
      <td>4.9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Love Always</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Explosions Now</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>4.8</td>
      <td>0.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Guns and Karate</td>
      <td>4.9</td>
      <td>NaN</td>
      <td>4.9</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Ninjas vs Cowboys</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>4.8</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Paint Drying</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.1</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




The above table is showing us our "database". As we can see, Kitson has yet to watch "Two Lovers" as it is filled with a `NaN`. Our main job is to be able to predict what each of the `NaN`'s should be.

Let's first assume that each user provides us with their list of preferences for movies. Let's first say that Kitson really likes romantic movies and really hates action movies. We can represent this via:

`beta_Kitson` = [4.5, 1]

In the equation above, the first number represents, on a scale of 0 to 5, how much Kitson likes romantic movies. Similarily the second number represents how much Kitson likes action movies.

Next, let's assume that each movie also has a "genre score" saying how romantic or action packed a movie is. We can do this via:

`x_twolovers` = [0.95, 0.01]

In the above equation, the first number represents how romantic the movie is on a scale of 0 to 1. The same is true for the second value on measuring how action packed a movie is.

It should be noted that the assumptions of knowing `beta` and `x` will eventually be discarded. For now however, let's assume that all users have given us their preferences, `beta_user`, and each director has given us a genre score for each movie `x_movie`

Next, we can find the dot product of `x_twolovers` and `beta_Kitson` to get Kitson's predicted score for "Two Lovers":

`x_twolovers`  dot  `beta_Kitson` = 4.285

Now, we can make a guess that Kitson will rate "Two Lovers" as a 4.285

Let's try implementing this:


```python
# Setting up Kitson's preferences and the movie "genre score"
beta_Kitson = np.array([4.5, 1])
x_twolovers = np.array([0.95, 0.01])

pred_Kitson_twolovers = np.dot(beta_Kitson, x_twolovers)
pred_Kitson_twolovers
```




    4.2849999999999993



Great, now let's try extending this for all users:


```python
# Setting up all movie's "genre-score":
x = np.array([[0.90,0.05], # Love Everywhere
              [0.95, 0.01], # Two Lovers
              [1,0], # Love Always
              [0.025,0.99], # Explosions Now
              [0.01,0.99], # Guns and Karate
              [0.02,0.95], # Ninjas vs Cowboys
              [0,0.1]]) # Paint Drying

# Setting up all user's preferenes:
beta = np.array([[1,5], # Matt
                 [4.5,1], # Kitson
                 [0,5], # Subi
                 [5,0], # Pratheek
                 [0.1,5], # Andrew
                 [5,0.1]]) # Rodrigo

# Determining predicted scores for all users and movies:
y_pred = np.dot(x, beta.T)
pd.DataFrame(y_pred.round(3))
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.150</td>
      <td>4.100</td>
      <td>0.25</td>
      <td>4.500</td>
      <td>0.340</td>
      <td>4.505</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.000</td>
      <td>4.285</td>
      <td>0.05</td>
      <td>4.750</td>
      <td>0.145</td>
      <td>4.751</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.000</td>
      <td>4.500</td>
      <td>0.00</td>
      <td>5.000</td>
      <td>0.100</td>
      <td>5.000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.975</td>
      <td>1.102</td>
      <td>4.95</td>
      <td>0.125</td>
      <td>4.953</td>
      <td>0.224</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.960</td>
      <td>1.035</td>
      <td>4.95</td>
      <td>0.050</td>
      <td>4.951</td>
      <td>0.149</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4.770</td>
      <td>1.040</td>
      <td>4.75</td>
      <td>0.100</td>
      <td>4.752</td>
      <td>0.195</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.500</td>
      <td>0.100</td>
      <td>0.50</td>
      <td>0.000</td>
      <td>0.500</td>
      <td>0.010</td>
    </tr>
  </tbody>
</table>
</div>





In the array shown above, each row corresponds to a specific movie and each column corresponds to a specific user. Concretely, the second row in the array corresponds to the movie "Two Lovers". The second entry in this row corresponds to Kitson's predicted rating of 4.3.

Now that we have predicted ratings for everyone and we can continue making all of our users happy by giving them good recommendations! However, we aren't quite done yet. If you haven't noticed already, there are errors in the array above. Take for example the second column corresponding to Kitson. Our predicted rating for the first movie "Love Everywhere" is 4.1. Howevever, if we look above Kitson actually provided a rating of 5 for this movie:


```python
y_raw
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Movie</th>
      <th>Matt</th>
      <th>Kitson</th>
      <th>Subi</th>
      <th>Pratheek</th>
      <th>Andrew</th>
      <th>Rodrigo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Love Everywhere</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>0.5</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Two Lovers</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.1</td>
      <td>4.9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Love Always</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Explosions Now</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>4.8</td>
      <td>0.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Guns and Karate</td>
      <td>4.9</td>
      <td>NaN</td>
      <td>4.9</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Ninjas vs Cowboys</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>4.8</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Paint Drying</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.1</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



We made an error of 0.9. If we continue cross referencing the predicted scores and the actual scores provided, we can find a lot of these errors:


```python
# Finding squared errors:
error = y_pred - y
init_sq_error = error**2
init_sq_error.round(2)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Matt</th>
      <th>Kitson</th>
      <th>Subi</th>
      <th>Pratheek</th>
      <th>Andrew</th>
      <th>Rodrigo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>0.81</td>
      <td>0.06</td>
      <td>0.25</td>
      <td>0.12</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.00</td>
      <td>NaN</td>
      <td>0.00</td>
      <td>0.06</td>
      <td>0.00</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.00</td>
      <td>0.25</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.00</td>
      <td>1.22</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.00</td>
      <td>NaN</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.05</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>0.81</td>
      <td>0.00</td>
      <td>0.09</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.25</td>
      <td>0.01</td>
      <td>NaN</td>
      <td>0.01</td>
      <td>0.25</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



The above table shows us the squared error associated with each predicted rating. The `NaN` values correspond to the case where a rating was not provided by the user so it is impossible to calculate an error. Let's see how much error we made for each user:


```python
init_user_errors = np.sum(init_sq_error)
init_user_errors
```




    Matt        2.307125
    Kitson      2.287106
    Subi        0.070000
    Pratheek    1.150625
    Andrew      0.405586
    Rodrigo     0.397828
    dtype: float64



Can we make the errors associated with each user smaller? Perhaps we can adjust our users `beta` preferences such that we achieve a smaller error. Alternatively, we can adjust our movie's `x` "genre-score" to achieve less error. Turns out we can do both simultaneously! Now, our new goal is to choose the value of `beta` and `x` such that we achieve the lowest possible error. This is a minimization problem. For the sake of clarity and brevity, I will skip over a lot of the mathematically dense theory and focus more on implementation.

First, we need a way of calculating the error:


```python
def compute_error(X_beta, y, rated, reg_coeff, num_features):
    # Get dimensions
    num_users = y.shape[1]
    num_movies = y.shape[0]

    # Reconstructing X:
    X = X_beta[0:num_movies*num_features]
    X = X.reshape((num_movies, num_features))

    # Reconstructing beta:
    beta = X_beta[num_movies*num_features:]
    beta = beta.reshape((num_users, num_features))

    # Calculating estimate:
    y_hat = np.dot(X, beta.T)

    # Calculating error:
    error = np.multiply((y_hat - y), rated)
    sq_error = error**2

    # Calculating cost:
    beta_regularization = (reg_coeff/2)*(np.sum(beta**2))
    X_regularization = (reg_coeff/2)*(np.sum(X**2))       
    J =  (1/2)*np.sum(np.sum(sq_error)) + beta_regularization + X_regularization

    # Calculating gradients:
    beta_gradient = np.dot(error.T,X) + reg_coeff*beta
    X_gradient = np.dot(error,beta) + reg_coeff*X
    X_beta_gradient = np.append(np.ravel(X_gradient), np.ravel(beta_gradient))

    return(J, X_beta_gradient)
```

#### Outputs of  `compute_error`:
Though seemingly intimidating, the function `compute_error` above is quite simple. Its main purpose is to compute `J` which is called the "cost". The "cost" is the thing that we are trying to minimize. "Cost" is pretty much the same thing as squared error and for the sake of this post, we can just think of "cost" and "squared error" to be the same thing.

The other output `X_beta_gradient` is the gradient of cost. In order to find the minimum of cost, we must take the derivative of cost with respective to `x` and `beta` separatley. `X_beta_gradient` can be thought of as the derivative of the cost function. For those who are interested in this, please [click here](https://en.wikipedia.org/wiki/Gradient_descent)

#### Inputs of  `compute_error`:
`X_beta` value is the genre-score and user preference arrays unrolled into a single vector array. This will be made more clear later.

`y` is matrix containing the ratings of each movie from each user.

`rated` is a boolean form of `y` showing whether or not a user has provided a rating for a specific movie

`reg_coeff` is the regularization constant. I will not be disussing this in this blog post as we are going to be setting this to zero.

`num_features` are the number of different features/genre scores we want associated with each movie. In the example done above, we only used two - romance and action. This can be any number of your choosing.

#### Magic Machine Learning Stuff:

Remember when I made the assumptions that we knew `x` and `beta`? Well now let's do away with those assumptions. Let's just make `x` and `beta` totally random and see if our algorithm can find optimal values for them.


```python
num_movies = y.shape[0]
num_users = y.shape[1]
num_features = 2 # romance and action

x = np.random.rand(num_users, num_features)
x
```




    array([[ 0.65182445,  0.87943029],
           [ 0.45902744,  0.30341172],
           [ 0.92346645,  0.74723157],
           [ 0.00874112,  0.68262371],
           [ 0.891005  ,  0.46805953],
           [ 0.58602187,  0.66814132]])




```python
beta = np.random.rand(num_movies, num_features)
beta
```




    array([[ 0.88522926,  0.13610893],
           [ 0.25787647,  0.26321224],
           [ 0.00373896,  0.30363864],
           [ 0.25413635,  0.3765358 ],
           [ 0.85459137,  0.73525599],
           [ 0.33363685,  0.24953996],
           [ 0.74352324,  0.43095807]])



Now that the values for `x` and `beta` are totally random ones. Let's now make the inputs that are required to use `compute_cost`:


```python
# Making X_beta:
X_beta = np.append(np.ravel(x), np.ravel(beta))
X_beta
```




    array([ 0.65182445,  0.87943029,  0.45902744,  0.30341172,  0.92346645,
            0.74723157,  0.00874112,  0.68262371,  0.891005  ,  0.46805953,
            0.58602187,  0.66814132,  0.88522926,  0.13610893,  0.25787647,
            0.26321224,  0.00373896,  0.30363864,  0.25413635,  0.3765358 ,
            0.85459137,  0.73525599,  0.33363685,  0.24953996,  0.74352324,
            0.43095807])



As mentioned above, `X_beta` is the unrolled version of the `x` matrix containing the values for the genre scores for each movie and the `beta` matrix containing the preferences for each user.


```python
# Making rated:
rated = ~pd.isnull(y)
rated
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Matt</th>
      <th>Kitson</th>
      <th>Subi</th>
      <th>Pratheek</th>
      <th>Andrew</th>
      <th>Rodrigo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6</th>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Setting remaining parameters:
reg_coeff = 0

# Let's try compute_error:
(J, grad) = compute_error(X_beta, y, rated, reg_coeff, num_features)
J
```




    160.06126386284626



The above is showing us that for our initial values of `x` and `beta` we have an "error" of 160. Are there different values of `x` and `beta` that will have less error? Of course there are! We just used random ones to start off with.

Next, we will be determining the values of `x` and `beta` that will minimize the "error" by using scipy's [`minimize`](https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.minimize.html) function combined with our `compute_error` function.

First however, we are going to center/normalize our ratings for each movie by subtracting the mean rating for each movie. By doing this, each movie will have a mean rating centered at zero. Although the reason why we do this is not that important or covered in this post, briefly, it is done so that we obtain reasonable results for users who have not yet rated anything.


```python
# Normalizing movie ratings across movies:
y_mean = np.mean(y,axis=1)
y_norm = y.T - y_mean.T
y_norm = y_norm.T
y_norm = y_norm.fillna(0)
y_norm.round(2)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Matt</th>
      <th>Kitson</th>
      <th>Subi</th>
      <th>Pratheek</th>
      <th>Andrew</th>
      <th>Rodrigo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00</td>
      <td>1.90</td>
      <td>-2.60</td>
      <td>1.90</td>
      <td>-3.10</td>
      <td>1.90</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-2.00</td>
      <td>0.00</td>
      <td>-2.00</td>
      <td>3.00</td>
      <td>-1.90</td>
      <td>2.90</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-2.50</td>
      <td>2.50</td>
      <td>-2.50</td>
      <td>2.50</td>
      <td>-2.50</td>
      <td>2.50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.52</td>
      <td>-2.48</td>
      <td>2.52</td>
      <td>-2.48</td>
      <td>2.32</td>
      <td>-2.38</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.94</td>
      <td>0.00</td>
      <td>1.94</td>
      <td>-2.96</td>
      <td>2.04</td>
      <td>-2.96</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2.54</td>
      <td>-1.46</td>
      <td>0.00</td>
      <td>-1.46</td>
      <td>2.34</td>
      <td>-1.96</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.02</td>
      <td>-0.02</td>
      <td>0.00</td>
      <td>0.08</td>
      <td>-0.02</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>



Now, we can use the `minimize` function in order to find the values of `x` and `beta` such that cost is minimized:


```python
min_results = minimize(fun=compute_error,
                       x0=X_beta,
                       method='CG',         
                       jac=True,
                       args=(y_norm, rated, reg_coeff, num_features),
                       options={'maxiter':1000})      
```

Our learned `x` and `beta` values will be stored in `min_results['x']`. We will need reconstruct `x` and `beta` by undoing our "unrolling" that we did earlier. Additionally, we must undo the normalization on the predicted `y` values that we did earlier:


```python
# Getting Results:
x_beta_learned = min_results['x']

# Reconstructing X:
x_learned = x_beta_learned[0:num_movies*num_features]
x_learned = x_learned.reshape((num_movies, num_features))

# Reconstructing beta:
beta_learned = x_beta_learned[num_movies*num_features:]
beta_learned = beta_learned.reshape((num_users, num_features))

# Undo normalization:
y_predicted_norm = np.dot(x_learned, beta_learned.T)
y_predicted = y_predicted_norm + y_mean[np.newaxis,:].T

# Making output pretty:
y_pred_df = pd.DataFrame(y_predicted)
y_pred_df['Movie']= y_raw['Movie']
y_pred_df.columns = np.append(y.columns, "Movie")
cols = y_pred_df.columns.tolist()
cols = cols[-1:] + cols[0:-1]
y_pred_df = y_pred_df[cols]
y_pred_df.round(2)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Movie</th>
      <th>Matt</th>
      <th>Kitson</th>
      <th>Subi</th>
      <th>Pratheek</th>
      <th>Andrew</th>
      <th>Rodrigo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Love Everywhere</td>
      <td>0.00</td>
      <td>4.93</td>
      <td>0.38</td>
      <td>4.94</td>
      <td>0.12</td>
      <td>5.13</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Two Lovers</td>
      <td>0.05</td>
      <td>5.06</td>
      <td>-0.03</td>
      <td>4.99</td>
      <td>0.07</td>
      <td>4.90</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Love Always</td>
      <td>-0.07</td>
      <td>5.01</td>
      <td>0.09</td>
      <td>4.98</td>
      <td>0.00</td>
      <td>5.03</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Explosions Now</td>
      <td>4.99</td>
      <td>0.02</td>
      <td>4.84</td>
      <td>0.05</td>
      <td>4.92</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Guns and Karate</td>
      <td>4.92</td>
      <td>-0.10</td>
      <td>5.00</td>
      <td>-0.04</td>
      <td>4.90</td>
      <td>0.06</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Ninjas vs Cowboys</td>
      <td>4.96</td>
      <td>0.90</td>
      <td>4.67</td>
      <td>0.89</td>
      <td>4.87</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Paint Drying</td>
      <td>-0.00</td>
      <td>0.05</td>
      <td>0.00</td>
      <td>0.05</td>
      <td>-0.00</td>
      <td>0.05</td>
    </tr>
  </tbody>
</table>
</div>



The table above is showing us the predicted ratings for each user.

## Discussion:

Let's compare the predicted ratings above with the raw input ratings that we've recieved from the users


```python
y_raw
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Movie</th>
      <th>Matt</th>
      <th>Kitson</th>
      <th>Subi</th>
      <th>Pratheek</th>
      <th>Andrew</th>
      <th>Rodrigo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Love Everywhere</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>0.5</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Two Lovers</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.1</td>
      <td>4.9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Love Always</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Explosions Now</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>4.8</td>
      <td>0.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Guns and Karate</td>
      <td>4.9</td>
      <td>NaN</td>
      <td>4.9</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Ninjas vs Cowboys</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>4.8</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Paint Drying</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.1</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



If we only look at the raw data and focus on Kitson's ratings, we can guess that he really likes romantic movies and does not like action movies. These preferences are reflected in the array containing the predicted ratings. We have predicted that he would give the movie "Two Lovers" a high rating while giving a movie like "Guns and Karate" a low rating.

#### More machine learning magic:

Something awesome has happened. At no point in time did we tell the algorithm to be trying to make features of genre scores for "Romance" and "Action". The algorithm has seemed to do this automatically. The algorithm has "learned" its own features by finding patterns in the dataset. As a concrete example, the algorithm essentially "notices" that users that have enjoyed "Love Everywhere" and "Love Always" tend to also enjoy "Two Lovers". From this, the algorithm adjusts the parameters in `x` to reflect that these three movie are similar.

Extending this logic, we can give this algorithm a very large database of movies and tell it to learn 20 features instead of just having two of "Romance" and "Action". In fact, these features may have nothing to do with genres at all. The algorithm could find that people who like actor "George Clooney" always rate his movies very highly and it would have this as a feature if it is what minimizes the cost most effectively.

This concludes this blog post. In future posts, I plan on taking this algorithm and applying it to a real movie dataset.
