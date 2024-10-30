import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

st.title("Sentiment Analysis of Tweets about US Airlines")
st.sidebar.title("Sentiment Analysis of Tweets about US Airlines")

st.markdown(" This application is a Streamlit app used to analyze the sentiment of the tweets 🐦 about US airlines ✈️ ")
st.sidebar.markdown(" This application is a Streamlit app used to analyze the sentiment of the tweets 🐦 about US airlines ✈️ ")


url = 'https://raw.githubusercontent.com/bkalyandheeraj2003/machine_learning-webapp/main/Tweets.csv'


@st.cache_data(persist=True)
def load_data():
    url = 'https://raw.githubusercontent.com/bkalyandheeraj2003/machine_learning-webapp/main/Tweets.csv'
    data = pd.read_csv(url)
    data['tweet_created'] = pd.to_datetime(data['tweet_created'])
    return data

data = load_data()

st.sidebar.subheader("Show random tweet")
random_tweet = st.sidebar.radio('Sentiment type', ('positive','negative','neutral'), key='tweet_radio')
st.sidebar.markdown(data.query('airline_sentiment == @random_tweet')[["text"]].sample(n=1).iat[0,0])

st.sidebar.markdown("### Number of tweets by sentiment type")
select = st.sidebar.selectbox('Visualization type', ['Histogram', 'Pie Chart'], key='viz_select')
sentiment_count = data['airline_sentiment'].value_counts()
sentiment_count = pd.DataFrame({'Sentiment': sentiment_count.index, 'Tweets': sentiment_count.values})

if not st.sidebar.checkbox('Hide', False, key='hide_tweets'):
    st.markdown("### Number of tweets by Sentiment")
    if select == "Histogram":
        fig = px.bar(sentiment_count, x='Sentiment', y='Tweets', color='Tweets', height=500)
        st.plotly_chart(fig)
    else:
        fig = px.pie(sentiment_count, values='Tweets', names='Sentiment')
        st.plotly_chart(fig)

st.sidebar.subheader("When and where are the users tweeting from?")
hour = st.sidebar.slider("Hour of day", 0, 23, key='hour_slider')
modified_data = data[data['tweet_created'].dt.hour == hour]
if not st.sidebar.checkbox("Close", False, key='close_map'):
    st.markdown("### Tweets location based on the time of the day")
    st.markdown("%i tweets between %i:00 and %i:00" % (len(modified_data), hour, (hour+1)%24))
    st.map(modified_data)
    if st.sidebar.checkbox("Show raw data", False, key='show_raw_data'):
        st.write(modified_data)

st.sidebar.subheader("Breakdown airline tweets by sentiment")
choice = st.sidebar.multiselect("Pick airlines", ('US Airways', 'United', 'American', 'Southwest', 'Delta', 'Virgin America'), key='airline_multiselect')

if len(choice) > 0:
    choice_data = data[data.airline.isin(choice)]
    fig_0 = px.histogram(choice_data, x='airline', y='airline_sentiment', histfunc='count', color='airline_sentiment', facet_col='airline_sentiment', labels={'airline_sentiment':'tweets'}, height=600, width=800)
    st.plotly_chart(fig_0)

st.sidebar.header("Word Cloud")
word_sentiment = st.sidebar.radio('Display word cloud for which sentiment?', ('positive', 'negative', 'neutral'), key='wordcloud_radio')

if not st.sidebar.checkbox("Close", False, key='close_wordcloud'):
    st.header('Word cloud for %s sentiment' % (word_sentiment))
    df = data[data['airline_sentiment'] == word_sentiment]
    words = ' '.join(df['text'])
    processed_words = ' '.join([word for word in words.split() if 'http' not in word and word.startswith('@') == False and word != 'RT'])
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', height=600, width=800).generate(processed_words)
    fig, ax = plt.subplots()

# Plot the word cloud on this specific figure/axis
    ax.imshow(wordcloud)

# Remove the axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

# Display the plot in Streamlit
    st.pyplot(fig)
