"""

@author: Akash

"""

import nltk
from gensim.models import Word2Vec
from nltk.corpus import stopwords

import re

passage = """APJ Abdul Kalam was among India’s best-known scientists before he became the country’s President. An alumnus of the Madras Institute of Technology, he worked for the Indian Space Research Organisation (ISRO) where he helped launch India’s first satellites into orbit. Later, Kalam worked on developing missiles and other strategic weapons; he was widely regarded as a national hero for leading India’s nuclear weapons tests in 1998. In 2002, Kalam was named the country’s President, and he held that position until 2007. During the Wharton India Economic Forum in Philadelphia, Kalam spoke with India Knowledge@Wharton about his career as a scientist, his vision for India’s future, and the most important traits for leaders, among other issues. An edited transcript of the interview follows:

India Knowledge@Wharton: Since our publication is called Knowledge@Wharton, could you tell us something about knowledge?


Kalam: I’ve written a four-line, poem-like thing called “Creativity.” It goes like this: “Learning gives creativity. Creativity leads to thinking. Thinking provides knowledge. Knowledge makes you great.” I have made at least a million children repeat these lines. I am very happy that Wharton has created Knowledge@Wharton; it’s a beautiful idea. My greetings to all of you.

India Knowledge@Wharton: Perhaps we could begin by talking about your own past. You were born in Rameswaram in 1931. What are the biggest differences between India as it was then and India today?

Kalam: Since then I have orbited the sun 76 times. I have seen when I was a young boy the Second World War coming to an end, and the effect of war and injuries. I saw India attain her freedom in August 1947; I saw the economic ascent phase of India [beginning in] 1991. I have worked with visionaries like Prof. Vikram Sarabhai. I have seen the green revolution, the white revolution, and the telecom revolution; I have also seen the growth of information and communication technologies (ICT), as well as India’s successes in the space program and self-sufficiency in strategic weaponry. These are some of the things I have witnessed. Of course, we have a long way to go. Since we have to bring smiles to the faces of more than one billion people, we have many challenges ahead.

India Knowledge@Wharton: After studying aeronautics at the Madras Institute of Technology, you were one of India’s top scientists at the Defense Research and Development Organisation (DRDO) and then at the Indian Space Research Organisation (ISRO). You helped launch several successful missiles, which led to your getting the nickname, “Missile Man.” What challenges were involved in getting this program going and leading it successfully?

Kalam: I worked for ISRO for about 20 years. My team and I worked to put India’s first satellite into space. Then our team took up the Integrated Guided Missile Development Program. These were youthful teams that worked with me, and they have gone on to take up much larger projects. These in turn have led to great value addition in areas such as technology, infrastructure and, above all, human resources.

One of the important lessons I learned in the space and missile program was not just how to handle success but how to deal with failure. Wharton is in the management environment. I would like young people to understand how they should manage failure. In any project you take up, you will face problems. These problems should not become the captain of the project chief; the project chief should be the captain of the problems and defeat the problems.

India Knowledge@Wharton: You were actively involved in India’s nuclear weapons tests in 1998. Could you tell us about that experience and the lessons you learned?

Kalam: The main lesson I learned was how multiple technical teams and departments of the government of India could work together for a great mission as an industrial partnership. It was a great experience.

India Knowledge@Wharton: You are known to be deeply spiritual. Did you ever feel conflicted, or guilty, about developing missiles and nuclear weapons? Why, or why not?

Kalam: I realize that for my country’s development, peace is essential. Peace comes from strength — because strength respects strength. That is how our weaponized missiles were born. You need strength to keep the nation peaceful, so that you can focus on the necessary developmental missions. That is how I see it."""

#preprocessing data

text = re.sub(r'\[[0-9]*\]',' ',passage)
text = re.sub(r'\s+',' ',text)
text = text.lower()
text = re.sub(r'\d',' ',text)
text = re.sub(r'\s+',' ',text)

#preparing sentences

sentences = nltk.sent_tokenize(text)
words = [nltk.word_tokenize(sentence) for sentence in sentences]

for i in range(len(sentences)):
    sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')]
    
#Training Word2Vec model
model = Word2Vec(sentences,min_count=1) #if word frequency <1, then don't chose

words = model.wv.vocab

#finding word vectors of a particular word
vector = model.wv['team']

#finding most similar words for a word
similar = model.wv.most_similar('team')

