import matplotlib.pyplot as plt
from wordcloud import WordCloud

def positive_review(norm_train_reviews):
    plt.figure(figsize=(10, 10))
    positive_text = norm_train_reviews[1]
    WC = WordCloud(width=1000, height=500, max_words=500, min_font_size=5)
    positive_words = WC.generate(positive_text)
    plt.imshow(positive_words, interpolation='bilinear')
    plt.show()

def negative_review(norm_train_reviews):
    plt.figure(figsize=(10, 10))
    negative_text = norm_train_reviews[8]
    WC = WordCloud(width=1000, height=500, max_words=500, min_font_size=5)
    negative_words = WC.generate(negative_text)
    plt.imshow(negative_words, interpolation='bilinear')
    plt.show()