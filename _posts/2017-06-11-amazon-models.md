
# Using Real Movie Data from Amazon

In this blog post, I will be explore the use of a collaborative filtering algorithm on a real movie dataset.

### Problem Description:

As discussed in my last post, we can make a recommender system using a collaborative filtering algorithm. This algorithm will use data obtained from [here](http://jmcauley.ucsd.edu/data/amazon/). Big thanks to Julian for giving me access to this data!

### Initialization:

First, we will need to load appropriate libraries and do a little light wrangling.


```python
# Importing
import feather
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from keras import backend as k
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
```

    Using TensorFlow backend.



```python
# Constants:
NUM_OF_REVIEWS = 5000
MIN_NUM_REVIEWS_MOVIE = 250
MIN_NUM_REVIEWS_PERSON = 10
```


```python
# Reading data
df = feather.read_dataframe("../results/movie_df.feather")
```

The wrangling done below will only get movies that have reviewed a minimum amount of times. Additionally, only users that have rated a minimum amount of movies will be selected. This was mostly done because if I hadn't, I would often get a lot of obscure movie titles that I haven't heard of before. 


```python
# Find number of reviewers per movie and 
# the number of movies reviewed by each person
n_reviews_movie_df = df.groupby('asin').reviewer_id.nunique()
n_reviews_person_df = df.groupby('reviewer_id').asin.nunique()

# Get list of movies that have been reviewed
# more than MIN_NUM_REVIEWS_MOVIE times and 
# users that have reviewed more than 
# MIN_NUM_REVIEWS_PERSON
popular_movies = list(n_reviews_movie_df[n_reviews_movie_df > MIN_NUM_REVIEWS_MOVIE].index)
critical_people = list(n_reviews_person_df[n_reviews_person_df > MIN_NUM_REVIEWS_PERSON].index)

# Filter dataframe to use only popular movies
# and critical people:
popular_df = df[df['asin'].isin(popular_movies)]
popular_df = popular_df[popular_df['reviewer_id'].isin(critical_people)]

# Shuffle data:
popular_df = shuffle(popular_df, random_state=42)

# Peak at data:
popular_df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>reviewer_id</th>
      <th>asin</th>
      <th>overall</th>
      <th>review_time</th>
      <th>title</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1393425</th>
      <td>A328S9RN3U5M68</td>
      <td>B00005JMFQ</td>
      <td>5.0</td>
      <td>1068681600</td>
      <td>Love Actually (Widescreen Edition)</td>
      <td>13.70</td>
    </tr>
    <tr>
      <th>288400</th>
      <td>A1NSHD4YCL5DV3</td>
      <td>0792833236</td>
      <td>4.0</td>
      <td>1063497600</td>
      <td>Raging Bull</td>
      <td>12.70</td>
    </tr>
    <tr>
      <th>913721</th>
      <td>A2R9J5LULVKF6T</td>
      <td>6304176287</td>
      <td>4.0</td>
      <td>998956800</td>
      <td>Willy Wonka &amp;amp; Chocolate Factory [VHS]</td>
      <td>3.99</td>
    </tr>
    <tr>
      <th>860416</th>
      <td>A335GUD1YBS31U</td>
      <td>6303617719</td>
      <td>5.0</td>
      <td>998438400</td>
      <td>French Kiss [VHS]</td>
      <td>0.99</td>
    </tr>
    <tr>
      <th>1512736</th>
      <td>A1TFH5Y9I9M3YN</td>
      <td>B00005QCYC</td>
      <td>4.0</td>
      <td>1005091200</td>
      <td>Jurassic Park Trilogy</td>
      <td>36.27</td>
    </tr>
  </tbody>
</table>
</div>



Great! We now have a nice little dataframe. It's nice to see that VHS seems to be alive and kicking. 


```python
# Reduce size of dataset (REMOVE LATER):
# The main issue occurs below during gradient descent.
# np.dot() takes a long time. Perhaps I can use PySpark's
# dot product. 
popular_df = popular_df.head(NUM_OF_REVIEWS)
```

The piece of code is a bit of an embarrassment. This will be mentioned again below, but the main reason for this was for convinence. Later, I will be taking the dot product between two matricies. If I use all of the data, this dot product will take a long time. One solution is to take advantage of distributed computing. Perhaps in my next blog post, I will explore this issue.


```python
# Only take relevant columns:
rating_df = popular_df[['reviewer_id', 'asin', 'overall']]

# Obtaining number of reviewers and products reviewed:
n_reviewers = len(rating_df.reviewer_id.unique())
n_products = len(rating_df.asin.unique())

# Print results:
print("There are", n_reviewers, "reviewers")
print("There are", n_products, "products reviewed")
```

    There are 3560 reviewers
    There are 1551 products reviewed


Great, we have reviewers and movies. FYI, the reason why I am saying "product" in my code is because I was previously using other amazon product data sets available [here](http://jmcauley.ucsd.edu/data/amazon/).

Next, we are working to construct a matrix where each row is a reviewer and each column is a movie. I initially construct it as a sparse array, which I think is the correct thing to do. Unfortunately, later I simply work with it as a normal dense array. If I were to repeat this exercise, I would have liked to continue to work with it as a sparse matrix. 


```python
# Creating map to map from reviewer id/asin(product id) to a number:
reviewer_map = dict(zip(np.unique(rating_df.reviewer_id), list(range(n_reviewers))))
product_map = dict(zip(np.unique(rating_df.asin), list(range(n_products))))
```


```python
# Obtain numbers(index in sparse matrix) associated with each id:
reviewer_index = np.array([reviewer_map[reviewer_id] for reviewer_id in rating_df.reviewer_id])
product_index = np.array([product_map[asin] for asin in rating_df.asin])
```


```python
# Obtain ratings to be put into sparse matrix:
ratings = np.array(rating_df.overall)
```


```python
# Create sparse X matrix:
X = coo_matrix((ratings, (reviewer_index, product_index)), shape=(n_reviewers, n_products))
```

Great, the sparse matrix `X` has now been constructed. Let's just double check that we did everything right:


```python
# Performing sanity check:
person_of_interest = reviewer_map['A328S9RN3U5M68']
product_of_interest = product_map['B00005JMFQ']

X.toarray()[person_of_interest, product_of_interest]
```




    5.0



Great, this result makes sense.

Now, the rating data is in a sparse matrix where the rows are users and the columns are movies. Note that all values of 0 for ratings are actually missing reviews. Reviewers cannot give something 0 stars. This is because `coo_matrix` by default fills in missing values with 0.

### Making our first model:

Now, let's use gradient descent to make our first model:


```python
# Set up parameters:
n_iters = 1000
alpha = 0.01
num_latent_features = 2

# Using dense array for now:
X_array = X.toarray()
```

The code below uses gradient descent to learn the correct values for `U` and `V`. `U` can be thought as an array where each row is associated with a specific person and each column represents some sort of learned preference of theirs. These learned preferences are the number of latent features. `V` can be thought of as an array where each column is associated with a specific movie and the rows are the learned quality of the movie. This learned quality of the movie corresponds to the learned preference of the movie. 

For example, let's first assume that one of "learned preference" happend to be a preference towards scary movies. The "learned quality" of the movie would be how scary the movie was. This is further explained in my last blog post. 

As an additional note, I am performing gradient descent a little differently than I had in my previous blog post. Essentially, these are both the same.


```python
# Randomly initialize:
U = np.random.randn(n_reviewers, num_latent_features) * 1e-5
V = np.random.randn(num_latent_features, n_products) * 1e-5

# Perform gradient descent:
for i in range(n_iters):
    # Obtain predictions:
    X_hat = np.dot(U, V)     
    
    if np.isnan(X_hat[person_of_interest, product_of_interest]):
        print("ERROR")
        break
        
    # Obtain residual
    resid = X_hat - X_array
    resid[np.isnan(resid)] = 0
    
    # Calculate gradients:
    dU = np.dot(resid, V.T) 
    dV = np.dot(U.T, resid)
    
    # Update values:
    U = U - dU*alpha
    V = V - dV*alpha
    
    # Output every 10% to make sure on the right track:
    if (i%(n_iters/10) == 0):
        print("Iteration:", i, 
              "   Cost:", np.sum(resid**2), 
              "   Rating of interest:", X_hat[person_of_interest, product_of_interest])
    
X_pred = np.dot(U, V)
    
```

    Iteration: 0    Cost: 93360.0000001    Rating of interest: 1.90893973813e-10
    Iteration: 100    Cost: 92540.3338637    Rating of interest: 0.14717131487
    Iteration: 200    Cost: 92486.3546961    Rating of interest: 1.19057516107
    Iteration: 300    Cost: 92474.6219165    Rating of interest: 1.96765864765
    Iteration: 400    Cost: 92472.5455087    Rating of interest: 2.10226001765
    Iteration: 500    Cost: 92471.4063733    Rating of interest: 2.02517075751
    Iteration: 600    Cost: 92469.9506994    Rating of interest: 1.8234530529
    Iteration: 700    Cost: 92467.961488    Rating of interest: 1.51107277851
    Iteration: 800    Cost: 92465.6479947    Rating of interest: 1.13093953942
    Iteration: 900    Cost: 92463.4816496    Rating of interest: 0.764126997336



```python
predicted_val = X_pred[person_of_interest, product_of_interest]
actual_val = X_array[person_of_interest, product_of_interest]
print("Actual Rating: ", actual_val)
print("Predicted Rating: ", predicted_val)
```

    Actual Rating:  5.0
    Predicted Rating:  0.477522679528


The above prediction is awful. This is because we didn't handle `nan` properly. The algorithm above is trying to account for all the zero values. The "signal" is being drowned out. To fix this is simple:

### Making our second model:


```python
# Replacing zeros with nan:
X_array[X_array==0] = np.nan

# Set up parameters:
n_iters = 1000
alpha = 0.01
num_latent_features = 2

# Initialize U and V randomly
U = np.random.randn(n_reviewers, num_latent_features) * 1e-5
V = np.random.randn(num_latent_features, n_products) * 1e-5

# Perform gradient descent:
for i in range(n_iters):        
    # Obtain predictions:
    X_hat = np.dot(U, V)     
    
    if np.isnan(X_hat[person_of_interest, product_of_interest]):
        print("ERROR")
        break
        
    # Obtain residual
    resid = X_hat - X_array
    resid[np.isnan(resid)] = 0
    
    # Calculate gradients:
    dU = np.dot(resid, V.T) 
    dV = np.dot(U.T, resid)
    
    # Update values:
    U = U - dU*alpha
    V = V - dV*alpha
    
    # Output every 10% to make sure on the right track:
    if (i%(n_iters/10) == 0):
        print("Iteration:", i, 
              "   Cost:", np.sum(resid**2), 
              "   Rating of interest:", X_hat[person_of_interest, product_of_interest])
    
# Make prediction:
X_pred = np.dot(U, V)

predicted_val = X_pred[person_of_interest, product_of_interest]
actual_val = X_array[person_of_interest, product_of_interest]
print("Actual Rating: ", actual_val)
print("Predicted Rating: ", predicted_val)
```

    Iteration: 0    Cost: 93360.0    Rating of interest: -1.06505550792e-10
    Iteration: 100    Cost: 50902.6127541    Rating of interest: 5.38285985849
    Iteration: 200    Cost: 7359.64612045    Rating of interest: 4.85884908364
    Iteration: 300    Cost: 897.066788056    Rating of interest: 4.87233382983
    Iteration: 400    Cost: 156.58492863    Rating of interest: 4.88973723326
    Iteration: 500    Cost: 75.6818835324    Rating of interest: 4.91750899441
    Iteration: 600    Cost: 38.0496747094    Rating of interest: 4.93817080819
    Iteration: 700    Cost: 13.0631882476    Rating of interest: 4.95164697125
    Iteration: 800    Cost: 10.1209025019    Rating of interest: 4.96158102444
    Iteration: 900    Cost: 8.1104596063    Rating of interest: 4.96938531597
    Actual Rating:  5.0
    Predicted Rating:  4.97574576552


This is a huge improvement! Awesome! Next, I will try to do some analysis on these results:

### Analysis:

One idea that can be done as a result from our model is we can group movies together. We have obtained these "learned features", why not use them? So now, I will try to make a plot to visualize the learned features:


```python
# Let's try to see which products are most similar to each other:
param1 = V.T[:,0]
param2 = V.T[:,1]
```


```python
# Plotting the products:
plt.plot(param1, param2, 'ro')
plt.show()
```


![png](output_32_0.png)


Interesting, when using 2 latent-features we seem to have movies that form a circular shape! I am not sure why, but it looks interesting! Now, let's try to cluster them together to see which movies are most similar to each other:


```python
# Performing k-means:
kmeans = KMeans(n_clusters=10, random_state=0).fit(V.T)
groups = kmeans.labels_

# Making plot:
plt.scatter(param1, param2, c=groups)
plt.show()
```


![png](output_34_0.png)


Looks like a sprinkled donut. It would be nice to see concrete movie titles associated with each of these groups. This is done below:


```python
# Making dictionary connecting asin to with title and group:
asin = np.unique(popular_df.asin)
product_dictionary = dict(zip(popular_df.asin, popular_df.title))
product_results = pd.DataFrame({'asin': asin})
product_results['title'] = [product_dictionary[x] for x in product_results.asin]
product_results['group'] = groups
```


```python
# Showing group 0:
product_results[product_results.group == 0].head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>asin</th>
      <th>title</th>
      <th>group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12</th>
      <td>0767802624</td>
      <td>Men in Black (Collector's Series)</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0767817478</td>
      <td>Godzilla</td>
      <td>0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>0767825411</td>
      <td>Ghostbusters [VHS]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>43</th>
      <td>0767853636</td>
      <td>Annie (Widescreen Edition)</td>
      <td>0</td>
    </tr>
    <tr>
      <th>51</th>
      <td>0780619609</td>
      <td>Teenage Mutant Ninja Turtles II - The Secret o...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Maybe this group is associated with movies that are action-y?


```python
# Showing group 1:
product_results[product_results.group == 1].head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>asin</th>
      <th>title</th>
      <th>group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>0767002652</td>
      <td>Upstairs Downstairs - The Premiere Season [VHS]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0767087372</td>
      <td>Upstairs Downstairs Collector's Edition</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0767736680</td>
      <td>Home Alone/Home Alone 2 Combo Pack [VHS]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>076780192X</td>
      <td>Close Encounters of the Third Kind (Widescreen...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0767809254</td>
      <td>Steel Magnolias [VHS]</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Not sure what this group is exactly, but I'm happy that it placed the two Upstairs Downstairs titles in the same group!


```python
# Showing group 2:
product_results[product_results.group == 2].head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>asin</th>
      <th>title</th>
      <th>group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>0767802535</td>
      <td>Big Night</td>
      <td>2</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0767803434</td>
      <td>Air Force One</td>
      <td>2</td>
    </tr>
    <tr>
      <th>33</th>
      <td>0767824407</td>
      <td>Immortal Beloved [VHS]</td>
      <td>2</td>
    </tr>
    <tr>
      <th>40</th>
      <td>0767839129</td>
      <td>The Messenger: The Story of Joan of Arc [VHS]</td>
      <td>2</td>
    </tr>
    <tr>
      <th>42</th>
      <td>0767851013</td>
      <td>Little Women (Collector's Series)</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



I really have no idea as to what is occurring in this group

### Closing Notes:

Unfortunately, these groups don't seem to make immediate sense. Honestly, the trends within groups that seem to be present are probably by luck. It would be nice to see groups of movies that had an obvious theme going on such as genre or time when released. Perhaps in the future, we can make better groups if we use more latent features (we used 2 in this case). 

In future blog posts I plan to:
* Scale up to use more data
* Create a twitter bot that may be able to identify which movies are trending on social media
    * This can be used to promote movies that are trending to people that we predicted would give it a high rating. 
    * The same idea can be applied to Amazon's "Toys and Games" data set. My thought that it would be able to promote something like a fidget spinner to people which seems to be all the rage now a days. 
* Create better models be adding more features to our dataset. This can easily be done by altering our gradient descent loop. 

Please let me know if you have any other cool ideas/questions/suggestions/**found mistakes**! My e-mail is andrewlim90@gmail.com
