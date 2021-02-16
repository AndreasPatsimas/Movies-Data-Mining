from sklearn.feature_extraction.text import TfidfVectorizer

def execute_tf_idf(normalized_reviews_train, normalized_reviews_test):

    tv = TfidfVectorizer(min_df=0, max_df=1, use_idf=True, ngram_range=(1, 3))

    tfidf_vect_reviews_train = tv.fit_transform(normalized_reviews_train)

    tfidf_vect_reviews_test = tv.transform(normalized_reviews_test)
    print('TFIDF_Vectorizer_reviews_train:', tfidf_vect_reviews_train.shape)
    print('TFIDF_Vectorizer_reviews_test:', tfidf_vect_reviews_test.shape)

    my_dictionary = {'tfidf_vect_reviews_train': tfidf_vect_reviews_train, 'tfidf_vect_reviews_test': tfidf_vect_reviews_test}

    return my_dictionary