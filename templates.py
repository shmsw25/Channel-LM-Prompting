TEMPLATES = {
    'climate_fever': [
        ("We know that %s", "We are %s that humans are causing global warming. ", ["sure", "against", "unsure", "hesitated"] ),
        ("%s\\n\\nCan we conclude that humans are causing global warming?\\n\\n", "%s. ", ["yes", "no", "maybe", "disputed"]),
        ("%s\\n\\nIs it true that humans are causing global warming?\\n\\n", "%s. ", ["yes", "no", "maybe", "disputed"]),
        ("%s\\n\\nAre humans causing global warming?\\n\\n", "%s. ", ["yes", "no", "maybe", "disputed"]),
        ("%s\\n\\nWhat's the best answer to this question: Are humans causing global warming?\\n\\n", "%s. ", ["yes", "no", "maybe", "disputed"])
    ],
    'ethos-national_origin': [
        ("%s", "This is %s to national origin. ", ["related", "unrelated"]),
        ("%s\\n\\nIs it about national origin?\\n\\n", "%s. ", ["no", "yes"]),
        ("%s\\n\\nCan we conclude that it is about national origin?\\n\\n", "%s. ", ["no", "yes"]),
        ("%s\\n\\nWhat's the best answer to this question: is it about national origin?\\n\\n", "%s. ", ["no", "yes"]),
        ("Text: %s\\n\\nQuestion: Is it about national origin?\\n\\n", "%s. ", ["no", "yes"])
    ],
    'ethos-race': [
        ("%s", "This is %s to race. ", ["related", "unrelated"]),
        ("%s\\n\\nIs it about race?\\n\\n", "%s. ", ["no", "yes"]),
        ("%s\\n\\nCan we conclude that it is about race?\\n\\n", "%s. ", ["no", "yes"]),
        ("%s\\n\\nWhat's the best answer to this question: is it about race?\\n\\n", "%s. ", ["no", "yes"]),
        ("Text: %s\\n\\nQuestion: Is it about race?\\n\\n", "%s. ", ["no", "yes"])
    ],
    'ethos-religion': [
        ("%s", "This is %s to religion. ", ["related", "unrelated"]),
        ("%s\\n\\nIs it about religion?\\n\\n", "%s. ", ["no", "yes"]),
        ("%s\\n\\nCan we conclude that it is about religion?\\n\\n", "%s. ", ["no", "yes"]),
        ("%s\\n\\nWhat's the best answer to this question: is it about religion?\\n\\n", "%s. ", ["no", "yes"]),
        ("Text: %s\\n\\nQuestion: Is it about religion?\\n\\n", "%s. ", ["no", "yes"])
    ],
    'financial_phrasebank': [
        ("%s", "This is %s. ", ["great", "terrible", "okay"]),
        ("%s\\nWhat is the sentiment of this sentence?\\n", "%s. ", ["positive", "negative", "neutral"]),
        ("%s\\nHow would you describe the sentiment of this sentence?\\n", "%s. ", ["positive", "negative", "neutral"]),
        ("Determine the sentiment: \\n\\n%s\\n", "%s. ", ["positive", "negative", "neutral"]),
        ("%s\\nIs the sentiment of this sentence positive, negative, or neutral?\\n", "%s. ", ["positive", "negative", "neutral"])
    ],
    'hate_speech18' : [
        ("%s", "This is %s to hate speech. ", ["related", "unrelated"]),
        ("%s\\n\\nIs it hate speech?\\n\\n", "%s. ", ["yes", "no"]),
        ("%s\\n\\nCan we conclude that it is hate speech?\\n\\n", "%s. ", ["yes", "no"]),
        ("%s\\n\\nWhat's the best answer to this question: is it hate speech?\\n\\n", "%s. ", ["yes", "no"]),
        ("Text: %s\\n\\nQuestion: Is it hate speech?\\n\\n", "%s. ", ["yes", "no"])
    ],
    'medical_questions_pairs': [
        ("%s\\n%s\\n", "These two questions have %s meanings. ", ["similar", "different"]),
        ("%s\\n%s\\nWould you say that these questions are the same?\\n", "%s. ", ["yes", "no"]),
        ("%s\\n%s\\nDo those questions have the same meaning?\\n", "%s. ", ["yes", "no"]),
        ("%s\\n%s\\nAre these two questions inquiring about the same information?\\n", "%s. ", ["yes", "no"]),
        ("%s\\n%s\\nPlease tell me if those questions are the same.\\n", "%s. ", ["yes", "no"])
    ],
    'poem_sentiment': [
        ("%s", "This is %s. ", ["great", "terrible", "okay", "complicated"]),
        ("%s\\nWhat is the sentiment of this line?\\n", "%s. ", ["negative", "positive", "neutral", "mixed"]),
        ("%s\\nHow would you describe the sentiment of this line?\\n", "%s. ", ["negative", "positive", "neutral", "mixed"]),
        ("Determine the sentiment: \\n\\n%s\\n", "%s. ", ["negative", "positive", "neutral", "mixed"]),
        ("%s\\nIs the sentiment of this line negative, positive, neutral, or mixed?\\n", "%s. ", ["negative", "positive", "neutral", "mixed"])
    ],
    'superglue-cb': [
        ("Based on that %s, we conclude that %s", "This is %s. ", ["correct", "incorrect", "unclear"])
        ("%s\\n\\nBased on the paragraph above can we conclude that \"%s\"?\\n\\n", "%s. ", ["yes", "no", "maybe"]),
        ("%s\\n\\nBased on that paragraph can we conclude that this sentence is true?\\n%s\\n\\n", "%s. ", ["yes", "no", "maybe"]),
        ("%s\\n\\nCan we draw the following conclusion?\\n%s\\n\\n", "%s. ", ["yes", "no", "maybe"]),
        ("%s\\n\\nDoes this next sentence follow, given the preceding text?\\n%s\\n\\n", "%s. ", ["yes", "no", "maybe"])
    ],
    'tweet_eval-hate': [
        ("%s", "This is %s to hate speech. ", ["unrelated", "related"]),
        ("%s\\n\\nIs it hate speech?\\n\\n", "%s. ", ["no", "yes"]),
        ("%s\\n\\nCan we conclude that it is hate speech?\\n\\n", "%s. ", ["no", "yes"]),
        ("%s\\n\\nWhat's the best answer to this question: is it hate speech?\\n\\n", "%s. ", ["no", "yes"]),
        ("Text: %s\\n\\nQuestion: Is it hate speech?\\n\\n", "%s. ", ["no", "yes"])
    ],
    'tweet_eval-stance_atheism': [
        ("%s", "This statement %s atheism. ", ["ignores", "hates", "favors"]),
        ("%s\\n\\nDoes it favor atheism?\\n\\n", "%s. ", ["maybe", "no", "yes"]),
        ("%s\\n\\nCan we conclude that it favors atheism?\\n\\n", "%s. ", ["maybe", "no", "yes"]),
        ("%s\\n\\nWhat's the best answer to this question: does it favor atheism?\\n\\n", "%s. ", ["maybe", "no", "yes"]),
        ("Text: %s\\n\\nQuestion: Does it favor atheism?\\n\\n", "%s. ", ["maybe", "no", "yes"])
    ],
    'tweet_eval-stance_feminist': [
        ("%s", "This statement %s feminist. ", ["ignores", "hates", "favors"]),
        ("%s\\n\\nDoes it favor feminist?\\n\\n", "%s. ", ["maybe", "no", "yes"]),
        ("%s\\n\\nCan we conclude that it favors feminist?\\n\\n", "%s. ", ["maybe", "no", "yes"]),
        ("%s\\n\\nWhat's the best answer to this question: does it favor feminist?\\n\\n", "%s. ", ["maybe", "no", "yes"]),
        ("Text: %s\\n\\nQuestion: Does it favor feminist?\\n\\n", "%s. ", ["maybe", "no", "yes"])
    ],
}