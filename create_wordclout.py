#!/usr/bin/env python
# encoding: utf-8

from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import csv 

infile = "data/ArtistDiversityFeedbackWords_20200926.csv"
  
comment_words = '' 
stopwords = set(STOPWORDS) 
  
# iterate through the csv file 
with open (infile, 'r') as inf:
    _reader = csv.reader(inf, delimiter=',')
    for row in _reader:
        word = [x.strip().replace(' ','') for x in row]
        comment_words += word[0] + " "

      

              
    wordcloud = WordCloud(width = 800, height = 800, 
                    background_color ='white', 
                    stopwords = stopwords, 
                    min_font_size = 10).generate(comment_words) 
      
    # plot the WordCloud image                        
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
      
    plt.show()  