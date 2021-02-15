def split(my_data):

    train_reviews = my_data.review[:40000]
    train_sentiments = my_data.sentiment[:40000]

    test_reviews = my_data.review[40000:]
    test_sentiments = my_data.sentiment[40000:]
    print(train_reviews.shape, train_sentiments.shape)
    print(test_reviews.shape, test_sentiments.shape)