---
title:  "Manual Linear Regression"
date:   2016-01-08 15:04:23
categories: [jekyll]
tags: [jekyll]
---
Often, I find that people rely heavily on packages/tools to perform there analysis. While abstraction is great, people often forget or lose sight of fundamentals. This post aims to redevelop some linear regression fundamentals. So, let's get some fundamentals down by doing manual linear regression rather than using `lm()`!

Setup:
------

Let's work with the baseball dataset:

#### Loading libraries:

``` r
rm(list = ls())
library(Lahman)
library(tidyverse)
library(broom)
```

#### Getting data:

Let's start with 1 explanatory variable (hits) and 1 response (homeruns):

``` r
h_hr <- Batting %>%
  filter(yearID >= 2014) %>%
  select(H, HR)

head(h_hr)
```

    ##     H HR
    ## 1   0  0
    ## 2  33  1
    ## 3 176 36
    ## 4   0  0
    ## 5   0  0
    ## 6   0  0

Let's visualize this:

``` r
ggplot(h_hr, aes(H, HR)) +
  geom_point()
```

![](https://raw.githubusercontent.com/AndrewLim1990/manual_regression/master/src/manual_regression_files/figure-markdown_github/unnamed-chunk-3-1.png)

Alright - let's work on putting a line through this.

Regression:
-----------

The code below gets our beta values.

-   This is taking the derivative of the error hat (residuals) and equating it to zero
-   By doing this, we are finding betas such that our residuals are minimized
-   Reminder: residuals are the distance from our line to the points
-   Click [here](https://github.ubc.ca/ubc-mds-2016/DSCI_561_regr-1_students/blob/master/lectures/lect07_diagnostics.pdf) (SLIDE 6) for proof

``` r
# How to get betas:
getBetas <- function(X, y) {
  betas <- solve(t(X)%*%X) %*% t(X) %*% y # https://github.ubc.ca/ubc-mds-2016/DSCI_561_regr-1_students/blob/master/lectures/lect07_diagnostics.pdf (SLIDE 6)
  return(betas)
}
```

Let's try it out:

``` r
# initializing:
n <- length(h_hr$H) # number of data points
X <- matrix(c(rep(1, n), h_hr$H), nrow = n, ncol = 2) # adding column of 1's for intercept
p <- dim(X)[2]
y <- h_hr$HR # our y0values

# get betas
betas <- getBetas(X=X, y=y)
betas
```

    ##            [,1]
    ## [1,] -0.1045731
    ## [2,]  0.1123100

Awesome - we've got some sort of values. Let's trying graphing them:

``` r
y_pred <- unlist(X %*% betas) #these are our predicted values

h_hr <- h_hr %>%
  mutate(y_hat = as.vector(y_pred))

ggplot(h_hr, aes(H, HR)) +
  geom_point() +
  geom_point(aes(H, y_hat, color='red'))
```

![](https://raw.githubusercontent.com/AndrewLim1990/manual_regression/master/src/manual_regression_files/figure-markdown_github/unnamed-chunk-6-1.png)

Let's now find our residuals:

``` r
h_hr <- h_hr %>%
  mutate(residuals = HR - y_hat)

head(h_hr)
```

    ##     H HR      y_hat  residuals
    ## 1   0  0 -0.1045731  0.1045731
    ## 2  33  1  3.6016564 -2.6016564
    ## 3 176 36 19.6619843 16.3380157
    ## 4   0  0 -0.1045731  0.1045731
    ## 5   0  0 -0.1045731  0.1045731
    ## 6   0  0 -0.1045731  0.1045731

Let's plot the residuals against our 'X' aka H:

``` r
ggplot(h_hr, aes(H, residuals)) +
  geom_point()
```

![](https://raw.githubusercontent.com/AndrewLim1990/manual_regression/master/src/manual_regression_files/figure-markdown_github/unnamed-chunk-8-1.png)

Comparison with `lm()`
----------------------

Now lets compare with what lm gives...

#### Betas matching:

``` r
model <- lm(HR~H, data=h_hr)
tidy(model)$estimate
```

    ## [1] -0.1045731  0.1123100

``` r
betas
```

    ##            [,1]
    ## [1,] -0.1045731
    ## [2,]  0.1123100

Awesome - our beta values match. How about the rest?

#### Std.error matching:

``` r
getBetaVar <- function(X, y){
  # do stuff from before:
  betas <- solve(t(X)%*%X) %*% t(X) %*% y
  y_hat <- as.vector(unlist(X %*% betas))
  residuals <- y-y_hat

  # get params:
  n <- dim(X)[1]
  p <- dim(X)[2]

  # finding variance:
  sigma_hat <- sqrt(sum((residuals^2))/(n - p))

  # version 1:  (matches)
  #s_x <- sd(X[,2])
  #beta_var <- sigma_hat/(sqrt(n-1)*s_x)

  # version 2: (does not match)
  cov_matrix <- sigma_hat^2*solve(t(X) %*% X)
  beta_var <- diag(cov_matrix)

  # version 3: (matches)
  #x_bar <- colMeans(X)
  #x_bar_matrix <- matrix(rep(x_bar, n), nrow=n, ncol=p, byrow=TRUE)
  #SSx <- sum((X - x_bar_matrix)^2)
  #beta_var <- sigma_hat^2/SSx

  beta_var
}

beta_var <- getBetaVar(X,y)

sqrt(beta_var)
```

    ## [1] 0.079728630 0.001434352

``` r
tidy(model)$std.error
```

    ## [1] 0.079728630 0.001434352

Awesome it worked. Let's try making our confidence interval:

``` r
beta0_se <- sqrt(beta_var)[1]
beta1_se <- sqrt(beta_var)[2]
t_val <- qt(p = 0.975, df = n-p)

# manually calculate confidence intervals:
data.frame(betas) %>%
  mutate(se = c(beta0_se, beta1_se),
         lower=betas-t_val*se,
         upper=betas+t_val*se)
```

    ##        betas          se      lower      upper
    ## 1 -0.1045731 0.079728630 -0.2609032 0.05175695
    ## 2  0.1123100 0.001434352  0.1094975 0.11512243

``` r
# use package to calculate confidence intervals:
confint_tidy(model)
```

    ##     conf.low  conf.high
    ## 1 -0.2609032 0.05175695
    ## 2  0.1094975 0.11512243

Nice our confidence intervals match.

[jekyll]:      http://jekyllrb.com
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-help]: https://github.com/jekyll/jekyll-help
