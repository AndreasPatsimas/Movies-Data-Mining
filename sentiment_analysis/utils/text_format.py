from bs4 import BeautifulSoup
import re

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

def remove_text_noise(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text