from sklearn.feature_extraction.text import TfidfVectorizer

def execute_tf_idf(norm_train_reviews, norm_test_reviews):
    # Tfidf vectorizer
    tv = TfidfVectorizer(min_df=0, max_df=1, use_idf=True, ngram_range=(1, 3))
    # transformed train reviews
    tv_train_reviews = tv.fit_transform(norm_train_reviews)
    # transformed test reviews
    tv_test_reviews = tv.transform(norm_test_reviews)
    print('Tfidf_train:', tv_train_reviews.shape)
    print('Tfidf_test:', tv_test_reviews.shape)

    my_dictionary = {'tv_train_reviews': tv_train_reviews, 'tv_test_reviews': tv_test_reviews}

    return my_dictionary