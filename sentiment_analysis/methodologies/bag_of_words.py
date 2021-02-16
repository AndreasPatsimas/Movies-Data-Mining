from sklearn.feature_extraction.text import CountVectorizer

def execute_bag_of_words(normalized_reviews_train, normalized_reviews_test):

    cv = CountVectorizer(min_df=0, max_df=1, binary=False, ngram_range=(1, 3))

    countvect_reviews_train = cv.fit_transform(normalized_reviews_train)

    countvect_reviews_test = cv.transform(normalized_reviews_test)

    print('Bow-Counter-Vectorizer-train:', countvect_reviews_train.shape)
    print('Bow-Counter-Vectorizer-test:', countvect_reviews_test.shape)

    my_dictionary = {'countvect_reviews_train': countvect_reviews_train, 'countvect_reviews_test': countvect_reviews_test}

    return my_dictionary