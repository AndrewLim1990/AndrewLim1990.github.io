This blog post will follow my progress on making a twitter bot to try and obtain movie titles that are "hot" or "trending" on social media in order to identify **Amazon products that can be promoted** to customers.

In the spirit of full disclosure, this blog post was fully inspired by [this Amazon job posting](https://www.amazon.jobs/en/jobs/515195)

Initialization
--------------

``` r
# Clearing workspace:
rm(list=ls())

# Loading libraries:
suppressMessages(library(tidyverse))
suppressMessages(library(stringr))
suppressMessages(library(feather))
suppressMessages(library(tm))
suppressMessages(library(roxygen2))
suppressMessages(library(knitr))
```

``` r
# Reading in data:
tweets <- suppressMessages(read_csv("../data/candidate_tweets.csv"))
movies <- suppressMessages(read_feather("../data/prediction_df.feather"))

# Peaking at tweets:
kable(head(tweets, 4) %>%
  select(name, text))
```

| name                 | text                                                                                                                                           |
|:---------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------|
| IMDb                 | Don't miss our Editor's picks for the week of June 11 👉                                                                                        |
| <https://t.co/VyufT> | Srej4 <https://t.co/HKcwtMA0nq>                                                                                                                |
| Yahoo Movies         | \#47MetersDown reviews are in: @Variety calls shark arcfor \#MandyMoore, \#ClaireHolt formulaic but effective… <https://t.co/Qr5nTxuXbg>       |
| Conan O'Brien        | Dear Mr. Bezos: There are less expensive ways to check out women in yoga pants.                                                                |
| Rotten Tomatoes      | It's a big \#Batman week, so we're ranking all of the Caped Crusader's movies from worst to best by 🍅 \#Tomatometer… <https://t.co/Kf1z7oUW8I> |

The twitter data above consists of tweets by movie focused twitter accounts within the last 24 hours. This data is obtained using a twitter bot that surveys the tweets every 24 hours. The code for the bot can be seen [here](https://github.com/AndrewLim1990/nlp_movie_tweets). It should be noted that I forked this repository from my friend [David's repository](https://github.com/laingdk/nlp_news_tweets) who used his twitter bot to identify "hot" news stories.

``` r
# Set seed for reproducability:
set.seed(16)

# Peaking at movies:
kable(head(movies[sample(nrow(movies)),]) %>%
       select(product_name, predicted_rating, actual_rating))
```

| product\_name                                                           |  predicted\_rating|  actual\_rating|
|:------------------------------------------------------------------------|------------------:|---------------:|
| I, Robot (Widescreen Edition)                                           |           3.747327|               3|
| V for Vendetta Special Edition                                          |           3.595076|               5|
| Event Horizon \[VHS\]                                                   |           3.356819|               1|
| On the Waterfront (includes Oscar's Greatest Moments 1971-1991) \[VHS\] |           4.961710|               5|
| Outfoxed - Rupert Murdoch's War on Journalism                           |           2.071590|               1|
| Coming to America \[VHS\]                                               |           5.310614|               5|

The data shown above was obtained from [my last blog post](https://andrewlim1990.github.io/using-real-movie-data-from-amazon/) where I implemented a colaborative filtering algorithm to predict ratings that people would give movies they haven't seen before.

How to use this data:
---------------------

My general idea is to identify which of the movies within our twitter dataset is being tweeted about the most. After identifying which movies are popular within social media, we can identify individuals that have a high predicted score for the movie and promote these products to them.

My first attempt at this will be very simplistic where I **very** crudely and naively vectorize each movie title into individual words and count the occurrences of each word within the twitter data set. I then divide the number of total occurrences by the number of words within the title. For example, if our movie title is "Home Alone", we will vectorize this title into two words "home" and "alone". Then, if "home" occurs in our data set 4 times and "alone" occurs 5 times, we have 9 total occurrences. I then will divide the total occurrences by the number of words in the title (2 in this case). This gives us 4.5 occurrences per word.

Once again, this is a **very crude** way of doing this. However, as a first attempt, it can serve as a proof of concept.

Data Wrangling:
---------------

Now, to get started, I need to clean the data. I'll start by defining some functions:

``` r
# Defining some functions

# Cleans text by removing stop words as well as commonly used/non-important words in movie tweets and movie titles
get_clean_text <- function(x){
  words_to_remove <- c(stopwords('en'), "vhs", "dvd", "edition", "widescreen", "combo pack", "special edition",
                     "season", "collector's edition", "full screen", "movie", "1", "2", "3", "4", "5", "6", "7",
                     "8", "9", "collector", "series", "amp", "film", "new", "like", "director", "show")
  stopwords_regex = paste(words_to_remove, collapse = '\\b|\\b')
  stopwords_regex = paste0('\\b', stopwords_regex, '\\b')

  movie_titles_cleaned <- tolower(x)
  movie_titles_cleaned = stringr::str_replace_all(movie_titles_cleaned, stopwords_regex, '')

  movie_titles_cleaned <- gsub('[[:punct:] ]+',' ',movie_titles_cleaned)

  movie_titles_cleaned <- str_replace(movie_titles_cleaned, " s ", " ")

  return(movie_titles_cleaned)
}

# Ranks each element of vector and assigning the same rank in the case of ties
get_rank <- function(x){
  x_unique <- unique(x)
  ranks <- rank(x_unique)
  ranked <- ranks[match(x, x_unique)]
  return(ranked)
}
```

Now the actual cleaning:

``` r
# Getting movie titles:
movie_titles <- unique(movies$product_name)

# Cleaning tweets
tweets_cleaned <- get_clean_text(tweets$text)
tweets_cleaned <- sapply(str_split(tweets_cleaned, "https"), '[[', 1) # remove hyper links from tweets
tweets <- tweets %>%
  mutate(text_cleaned = tweets_cleaned)

# Cleaning movie titles:
movie_titles_cleaned <- get_clean_text(movie_titles)

# Vectorizing movie titles:
movies_split <- str_split(movie_titles_cleaned, " ", simplify = TRUE)
movies_split[nchar(movies_split) <= 2] <- NA
movies_listed <- apply(movies_split, 1, function(x) unique(x[!is.na(x)]))
```

Finding "hot" movies
--------------------

As discussed above, I will now get the number of occurrences per word for each title:

``` r
# Making list:
total_occurrences_list <- c()

# Getting raw number occurrences for each movie title:
for(movie_title_split in movies_listed){
  total_occurrences = 0
  for(word in movie_title_split){
    regex_pattern <- paste0('\\s', word, '\\s')
    num_occurrences = sum(str_detect(string=tweets_cleaned, pattern=regex_pattern))
    total_occurrences = total_occurrences + num_occurrences
  }
  total_occurrences_list = c(total_occurrences_list, total_occurrences)
}

# Getting occurrences per word:
occurrence_per_word <- total_occurrences_list/lengths(movies_listed)

# Making a dataframe:
result_df <- data.frame(movie_title = movie_titles,
                        occurrences_per_word = occurrence_per_word) %>%
  mutate(words = movies_listed) %>%
  arrange(desc(occurrences_per_word))

# Displaying dataframe:
kable(head(result_df, 10))
```

| movie\_title                      |  occurrences\_per\_word| words           |
|:----------------------------------|-----------------------:|:----------------|
| The Game \[VHS\]                  |                     3.0| game            |
| For Love of the Game              |                     2.5| love, game      |
| The Long, Long Trailer \[VHS\]    |                     2.5| long, trailer   |
| Apocalypse Now \[VHS\]            |                     2.0| apocalypse, now |
| Batman - The Movie \[VHS\]        |                     2.0| batman          |
| They Live                         |                     2.0| live            |
| Best in Show                      |                     2.0| best            |
| Men in Black (Collector's Series) |                     1.5| men, black      |
| As Good As It Gets                |                     1.5| good, gets      |
| Groundhog Day                     |                     1.5| groundhog, day  |

Sanity Checks:
--------------

Already, some of the results shown above seem a bit suspect. Let's check them by viewing the tweets linked to the movies:

``` r
# Checking tweets involving: "The Game [VHS]":
for(word in result_df[1,]$words[[1]]){
  regex_pattern <- paste0('\\s', word, '\\s')
  detected_tweets <- str_detect(string=tweets_cleaned, pattern=regex_pattern)
  print(tweets_cleaned[detected_tweets])
}
```

    ## [1] "rt insideedgeamzn time witness game behind game insideedgetrailer now insideedge amazonvideoin excelmovies faroutak "
    ## [2] " gt game thrones jon snow real name revealed gt s "                                                                  
    ## [3] " ign game e32017 \n\n"

Darn, turns out this was a false positive. Although these tweets have the word "game" in it, they are not actually talking about the move "The Game".

``` r
# Checking tweets involving: "The Long, Long Trailer [VHS]":
for(word in result_df[3,]$words[[1]]){
  regex_pattern <- paste0('\\s', word, '\\s')
  detected_tweets <- str_detect(string=tweets_cleaned, pattern=regex_pattern)
  print(tweets_cleaned[detected_tweets])
}
```

    ## character(0)
    ## [1] "movies ray donovan trailer poster showtime released "                  
    ## [2] " gt happy death day trailer groundhog day lot mur "                    
    ## [3] "find sparked 1967 riot detroit trailer featuring johnboyega "          
    ## [4] "watch trailer romantic comedy homeagain starring rwitherspoon arrived "
    ## [5] "lots daddies first trailer daddyshome \n"

Darn, another false positive. Many of these twitter accounts announce trailers, so of course a movie title with the word "trailer" would be considered to be "hot" by this crude method.

``` r
# Checking tweets involving: "Batman - The Movie [VHS]"
for(word in result_df[5,]$words[[1]]){
  regex_pattern <- paste0('\\s', word, '\\s')
  detected_tweets <- str_detect(string=tweets_cleaned, pattern=regex_pattern)
  print(tweets_cleaned[detected_tweets])
}
```

    ## [1] " big batman week re ranking caped crusader s movies worst best tomatometer "
    ## [2] " movies jaden smith made brilliantly awful batman music vide "

There we go! Batman once again pulls through for us. Sadly, this was the week Adam West passed away so I think people were tweeting Batman related stuff. Perhaps to celebrate his life and his accomplishments, Amazon could have a Batman promotion!

Closing Notes / To Do:
----------------------

**Note: this blog post is not yet finished and I plan on updating it whenever I make progress.**

So far, this exercise served as a good proof of concept to myself. Moving forward, I have a few ideas of how to make improvements:

#### 1. Reduce the amount of false positives:

-   Using the data we already have, we can construct a dataset consisting of movie titles with their corresponding matched tweets. We can then provide labels "correct" and "incorrect". Once we've obtained such a data set, we can try to perform logistic regression to better identify when movies and tweets should and shouldn't be linked. This would also involve feature engineering. We can construct and features such as:
    -   Number of matching/unmatching words in the title
    -   Number of letters within matching words in the title
    -   [String edit distances](https://en.wikipedia.org/wiki/Edit_distance)
    -   Etc.
-   Use fuzzy string matching between non-vectorized movie titles and tweets. This may give us a more robust way of identifying links between movie titles and tweets. This is closely related to string edit distances mentioned above.

#### 2. Set up bot:

-   I would like to add myself to the dataset and provide a couple of rating for movies I like and dislike. From here, I would be able to set up the bot to automatically notify me whenever a movie that it predicts I will like is being discussed on social media!
-   Scale out to use Amazon's non-movie related products such as electronics or books.

Stay tuned! This project is far from finished. Let me know if you have any suggestions or would like to contribute! My repository that I've been working out of can be found [here](https://github.com/AndrewLim1990/recommender_post2) for the recommender system and [here](https://github.com/AndrewLim1990/nlp_movie_tweets) for the NLP/twitter-bot part.
