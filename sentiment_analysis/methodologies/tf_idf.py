from sklearn.feature_extraction.text import TfidfVectorizer

def execute_tf_idf(norm_train_reviews, norm_test_reviews):

    tf_idf_vect = TfidfVectorizer(min_df=0, max_df=1, use_idf=True, ngram_range=(1, 3))

    tf_idf_vect_train_reviews = tf_idf_vect.fit_transform(norm_train_reviews)

    tf_idf_vect_test_reviews = tf_idf_vect.transform(norm_test_reviews)
    print('Tfidf_train:', tf_idf_vect_train_reviews.shape)
    print('Tfidf_test:', tf_idf_vect_test_reviews.shape)

    my_dictionary = {'tv_train_reviews': tf_idf_vect_train_reviews, 'tv_test_reviews': tf_idf_vect_test_reviews}

    return my_dictionary