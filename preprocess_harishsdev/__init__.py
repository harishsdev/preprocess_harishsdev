from preprocess_harishsdev import utils

__version__='0.0.2'


def get_word_counts(x):
    return  utils._get_word_counts(x)

def get_char_counts(x):
    return utils._get_char_counts(x)

def get_avg_wordlength(x):
    return utils._get_avg_wordlength(x)

def get_stopwords_counts(x):
    return utils._get_stopwords_counts(x)

def get_hashtag_counts(x):
    return utils._get_hashtag_counts(x)

def get_mentions_counts(x):
    return utils._get_mentions_counts(x)

def get_digit_counts(x):
    return utils._get_digit_counts(x)

 def get_uppercase_counts(x):
    return utils._get_uppercase_counts(x)

 def get_cont_to_exp(x):
    return utils._get_cont_to_exp(x)

def get_emails(x):
    return utils._get_emails(x)

 def remove_emails(x):
    return utils._remove_emails(x)

def get_urls(x):
    return utils._get_urls(x)

def remove_urls(x):
    return utils._remove_urls(x)

def remove_rt(x):
    return utils._remove_rt(x)

def remove_special_chars(x):
    return utils._remove_special_chars(x)

def remove_html_tags(x):
    return utils._remove_html_tags(x)

def remove_accented_chars(x):
    return utils._remove_accented_chars(x)

def make_to_base(x):
    return utils._make_to_base(x)




























