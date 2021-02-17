from sklearn.feature_extraction.text import CountVectorizer

def execute_bag_of_words(norm_train_reviews, norm_test_reviews):

    cv = CountVectorizer(min_df=0, max_df=1, binary=False, ngram_range=(1, 3))

    cv_train_reviews = cv.fit_transform(norm_train_reviews)

    cv_test_reviews = cv.transform(norm_test_reviews)

    print('BOW_cv_train:', cv_train_reviews.shape)
    print('BOW_cv_test:', cv_test_reviews.shape)

    my_dictionary = {'cv_train_reviews': cv_train_reviews, 'cv_test_reviews': cv_test_reviews}

    return my_dictionary