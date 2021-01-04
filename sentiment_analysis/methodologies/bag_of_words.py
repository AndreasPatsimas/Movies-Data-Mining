from sklearn.feature_extraction.text import CountVectorizer


# Bags of words model
# A bag-of-words model, or BoW for short, is a way of extracting features from text for use in modeling,
# such as with machine learning algorithms. ... A bag-of-words is a representation of text that describes
# the occurrence of words within a document. It involves two things: A vocabulary of known words.
# It is used to convert text documents to numerical vectors or bag of words.

def execute_bow(norm_train_reviews, norm_test_reviews):
    # Count vectorizer for bag of words
    cv = CountVectorizer(min_df=0, max_df=1, binary=False, ngram_range=(1, 3))
    # transformed train reviews
    cv_train_reviews = cv.fit_transform(norm_train_reviews)
    # transformed test reviews
    cv_test_reviews = cv.transform(norm_test_reviews)

    print('BOW_cv_train:', cv_train_reviews.shape)
    print('BOW_cv_test:', cv_test_reviews.shape)

    my_dictionary = {'cv_train_reviews': cv_train_reviews, 'cv_test_reviews': cv_test_reviews}

    return my_dictionary