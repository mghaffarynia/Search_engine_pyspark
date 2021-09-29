%sh 
pip install nltk
pip install --upgrade pip
python -m nltk.downloader all


!pip install nltk
import nltk
from nltk.corpus import stopwords
import string
import math

stop_words = set(stopwords.words('english'))

movie_names = sc.textFile('/FileStore/tables/movie_metadata.tsv')
movie_names = movie_names.map(lambda x: (x.split('\t')[0],x.split('\t')[2]))  # Using metadata for obtaining movie names

# Preparing data
summaries = sc.textFile("/FileStore/tables/plot_summaries.txt")
new_summaries = summaries.map(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
data = new_summaries.map(lambda x: x.split('\t'))
data = data.map(lambda x: (x[0], ' '.join([w for w in x[1].split() if w.lower() not in stop_words])))
data = data.flatMap(lambda x: (((x[0], i.lower()), 1) for i in x[1].split()))



doc_num = new_summaries.count()         # Total number of documents

total_number = data.map(lambda x: (x[0][1], 1)).reduceByKey(lambda x, y: x+y)
log_total = total_number.map(lambda x: (x[0],math.log(doc_num/x[1])))

# Computing tf-idf for each token in each document

freq = data.reduceByKey(lambda x,y:x+y)
tf = freq.map(lambda x: (x[0][1],(x[0][0],x[1])))
temp = tf.join(log_total)
values = temp.map(lambda x: ((x[1][0][0],x[0]),x[1][0][1]*x[1][1])).collect()
processed_data = sc.parallelize(values)

# Function for a single word query

def serach_engin_for_word(word, processed_data):
  print("Results for Top 10 movies related to '" + word + "' :")
  query = processed_data.filter(lambda x: x[0][1].lower() == word.lower())
  results = query.map(lambda x: (x[0][0],x[1])).join(movie_names).sortBy(lambda x: -x[1][0]).map(lambda x: x[1][1]).take(10)
  for res in results:
    print('\t', res, end='\n')
    
# Function for a sentence query
def searched_engin_for_sentence(sentence, processed_data):
  print("Results for Top 10 movies related to '" + sentence + "' :")

  # cleaning the sentence by removing punctuation from the sentence
  sentence = sentence.translate(str.maketrans('', '', string.punctuation))

  # getting rid of the stopwords from the sentence and converting the sentence to a list of words
  sentence = [word for word in sentence.lower().split(" ") if word not in stop_words]

  # converting the list of words to an rdd
  sentence_rdd = sc.parallelize(sentence)
  sentence_rdd = sentence_rdd.filter(lambda x: x!='')

  list_of_words = sentence_rdd.collect()

  # Computing idf of the sentence

  sentence_idf = sentence_rdd.map(lambda x: (x,1)).reduceByKey(lambda x,y : x+y).join(log_total)

  # Computing the term frequency for word
  sentence_tf = sentence_idf.map(lambda x: (x[0], x[1][0]))

  # Computing tf-idf for words
  sentence_tfidf = sentence_tf.join(sentence_idf).map(lambda x: (x[0],x[1][0]*x[1][1][1])).reduceByKey(lambda x,y: x+y)

  # preparing data for calculating cosine similarity
  sq_document = processed_data.map(lambda x:(x[0][0],x[1]*x[1])).reduceByKey(lambda x,y: x + y)
  sq_sentence_tfidf = sentence_tfidf.map(lambda x:(x[1]*x[1])).reduce(lambda x,y: x + y) 

  # finds the words that are in the sentence
  input = processed_data.filter(lambda x: x[0][1].lower() in list_of_words)

  # mapping the data and joining
  
  mapped_input = input.map(lambda x: (x[0][1],(x[0][0],x[1])))
  join_data = mapped_input.join(sentence_tfidf)
  semi_final = join_data.map(lambda x: (x[1][0][0], (x[1][0][1], x[1][1])))
  
  # data needed for computing the cosine similarity
  
  output = semi_final.join(sq_document)

  # Calculating the cosine similarity
  
  similarity = output.map(lambda x: (x[0], (x[1][0][0]*x[1][0][1])/(math.sqrt(x[1][1])*math.sqrt(sq_sentence_tfidf))))
  cosine = similarity.reduceByKey(lambda x,y: x+y)

  # Top 10 results for movies based on cosine similarity
  
  top_movies = cosine.sortBy(lambda x: -x[1])
  results = top_movies.join(movie_names).sortBy(lambda x: -x[1][0]).map(lambda x: x[1][1]).take(10)
  for res in results:
    print('\t', res, end='\n')
    
    
# The text file we uploaded for testing the searched engin is as below.
search_text = sc.textFile("/FileStore/tables/search.txt")   
for word in search_text.collect():
      searched_word = word.split(" ")
      if len(searched_word) == 1:
        
        serach_engin_for_word(word, processed_data)

      else:
        searched_engin_for_sentence(word, processed_data)














