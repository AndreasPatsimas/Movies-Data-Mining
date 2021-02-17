from sklearn.feature_extraction.text import CountVectorizer

def execute_bag_of_words(norm_train_reviews, norm_test_reviews):

    count_vect = CountVectorizer(min_df=0, max_df=1, binary=False, ngram_range=(1, 3))

    count_vect_train_reviews = count_vect.fit_transform(norm_train_reviews)

    count_vect_test_reviews = count_vect.transform(norm_test_reviews)

    print('BOW_count_vect_train:', count_vect_train_reviews.shape)
    print('BOW_count_vect_test:', count_vect_test_reviews.shape)

    my_dictionary = {'cv_train_reviews': count_vect_train_reviews, 'cv_test_reviews': count_vect_test_reviews}

    return my_dictionary