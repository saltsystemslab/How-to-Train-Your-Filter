"""
This file includes methods for encoding urls into vectors for model learning.

The feature extraction was taken from https://www.kaggle.com/code/bytadit/malicious-url-detection-with-ml-96-7-acc/notebook
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import urllib
import re
from sklearn.preprocessing import LabelEncoder
from urllib.parse import urlparse
from tld import get_tld


def vectorize_url(url : str):
    """
    Takes an input url and converts it into a vector based on its features.

    Parameters
    ----------
    url : str
        The url to convert into a vector.

    Returns
    -------
    A (np-array) vector encoding of the different features of the url.
    In order, the features are IP involvement, URL abnormality, dot count,
    www count, @ count, url depth, embeded domain count, suspicious url count,
    shortened url count, https count, http count, % count, - count, = count, URL length,
    hostname length, first directory length, tld length, digit count, and letter count.
    """
    ip_involvement = having_ip_address(url)
    url_abnormality = abnormal_url(url)
    subdomain_cnt = url.count('.')
    www_cnt = url.count('www')
    at_cnt = url.count('@')
    dir_cnt = no_of_dir(url)
    embedded_domains = no_of_embed(url)
    sus_urls = suspicious_words(url)
    shortened_urls = shortening_service(url)
    https_cnt = url.count('https')
    http_cnt = url.count('http')
    percent_cnt = url.count('%')
    dash_cnt = url.count('-')
    eq_cnt = url.count('=')
    url_len = len(url)
    host_len = len(urlparse(url).netloc)
    fd_len = fd_length(url)
    top_level_domain = tld_length(get_tld(url, fail_silently=True))
    number_count = digit_count(url)
    letter_cnt = letter_count(url)
    return np.array([ip_involvement, url_abnormality, subdomain_cnt, www_cnt, at_cnt, dir_cnt, embedded_domains,
                     sus_urls, shortened_urls, https_cnt, http_cnt, percent_cnt, dash_cnt,
                     eq_cnt, url_len, host_len, fd_len, top_level_domain, number_count, letter_cnt])

# check ip address usage
def having_ip_address(url):
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)' # IPv4 in hexadecimal
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # Ipv6
    if match:
        # print match.group()
        return 1
    else:
        # print 'No matching pattern found'
        return 0

# check if url is in abnormal format
def abnormal_url(url):
    hostname = urlparse(url).hostname
    hostname = str(hostname)
    match = re.search(hostname, url)
    if match:
        # print match.group()
        return 1
    else:
        # print 'No matching pattern found'
        return 0

# check directory depth
def no_of_dir(url):
    urldir = urlparse(url).path
    return urldir.count('/')

# check embedded domains
def no_of_embed(url):
    urldir = urlparse(url).path
    return urldir.count('//')

# check suspicious URL names
def suspicious_words(url):
    match = re.search('PayPal|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr',
                      url)
    if match:
        return 1
    else:
        return 0

# check shortened urls
def shortening_service(url):
    match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                      'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                      'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                      'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                      'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                      'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                      'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                      'tr\.im|link\.zip\.net',
                      url)
    if match:
        return 1
    else:
        return 0

# check first directory length
def fd_length(url):
    urlpath= urlparse(url).path
    try:
        return len(urlpath.split('/')[1])
    except:
        return 0

# check length of top-level domain
def tld_length(tld):
    try:
        return len(tld)
    except:
        return -1

# check number of numerical digits
def digit_count(url):
    digits = 0
    for i in url:
        if i.isnumeric():
            digits = digits + 1
    return digits

# check number of leters
def letter_count(url):
    letters = 0
    for i in url:
        if i.isalpha():
            letters = letters + 1
    return letters