label2int = {
    "fact": {"low": 0, "mixed": 1, "high": 2},
    "bias": {"extreme-left": 0, "left-center": 1, "left": 2, "center": 3, "right-center": 4, "right": 5, "extreme-right": 6},
}

int2label = {
    "fact": {0: "low", 1: "mixed", 2: "high"},
    "bias": {0: "extreme-left", 1: "left-center", 2: "left", 3: "center", 4: "right-center", 5: "right", 6: "extreme-right"},
}

TWITTER_ALL = "has_twitter,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified"
WIKI_ALL = "has_wikipedia,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc"
ARTICLE_ALL = "articles_body_glove,articles_title_glove"
ALEXA = "alexa"
ALL = ",".join([TWITTER_ALL, WIKI_ALL, ARTICLE_ALL, ALEXA])
FEATURE_MAPPING = {"TWITTER_ALL": TWITTER_ALL,
                   "WIKI_ALL": WIKI_ALL,
                   "ARTICLE_ALL": ARTICLE_ALL,
                   "ALEXA": ALEXA,
                   "ALL": ALL}

