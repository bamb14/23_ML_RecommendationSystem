import streamlit as st
import requests
from bs4 import BeautifulSoup
from html.parser import HTMLParser
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer


def get_user_data(user_name: str):
    url = 'https://fow.kr/find/'
    user_name = user_name.replace('#', '-')
    response = requests.request("GET", url + user_name)
    soup = BeautifulSoup(response.content, "html.parser")
    champion_list = []
    win_rate_list = []
    try:
        soup.select(
            '#content-container > div:nth-child(1) > div:nth-child(2) > div.rankchamp_S13B_div.rankchamp_S13B_div_all > table > tbody')[
            0].find_all('td')[(0)].get_text()  # 아이디 유효성 검사
    except:
        return champion_list, win_rate_list
    for i in range(5):
        try:
            champion_list.append(soup.select(
                '#content-container > div:nth-child(1) > div:nth-child(2) > div.rankchamp_S13B_div.rankchamp_S13B_div_all > table > tbody')[
                                     0].find_all('td')[(i * 15)].get_text().strip())  # 챔프0
            win_rate_list.append(int(soup.select(
                '#content-container > div:nth-child(1) > div:nth-child(2) > div.rankchamp_S13B_div.rankchamp_S13B_div_all > table > tbody')[
                                         0].find_all('td')[(i * 15) + 12].get_text()) / int(soup.select(
                '#content-container > div:nth-child(1) > div:nth-child(2) > div.rankchamp_S13B_div.rankchamp_S13B_div_all > table > tbody')[
                                                                                                0].find_all('td')[(
                                                                                                                              i * 15) + 1].get_text()))  # 게임 수 1
        except:
            break
    return champion_list, win_rate_list


# load matrix
user_matrix = pd.read_csv('user_matrix_mod.csv', index_col=0)
data = pd.read_csv('data_processed.csv')

user_matrix.reset_index(drop=True, inplace=True)


def predict(champion_list, win_rate_list):
    global user_matrix

    # make a similarity array
    user_matrix['similarity'] = 0.0

    # add input user's row
    user_matrix.loc['input'] = 0
    user_matrix.at['input', 'similarity'] = 1

    for j in range(5):
        user_matrix.at['input', champion_list[j]] = win_rate_list[j]

    # calculate similarity with pearson correlation
    user_matrix['similarity'] = user_matrix.corrwith(user_matrix.loc['input'], axis=1)

    # find 20 most similar users and make new matrix dataframe
    user_matrix = user_matrix.sort_values(by='similarity', ascending=False)
    user_matrix = user_matrix.iloc[:20]
    user_matrix.reset_index(drop=True, inplace=True)

    # do user-based collaborative filtering

    # calculate input user's row
    for i in range(len(user_matrix.columns) - 1):
        # sum of similarity, only users who have played the champion
        similarity_sum = 0.0
        for j in range(20):
            if user_matrix.iloc[j][user_matrix.columns[i]] != 0:
                similarity_sum += user_matrix.iloc[j]['similarity']
        if similarity_sum != 0:
            user_matrix.at['input', user_matrix.columns[i]] = (
                    np.dot(user_matrix.iloc[:20][user_matrix.columns[i]],
                           user_matrix.iloc[:20]['similarity']) / similarity_sum)

    result = []
    result_score = []
    # sort user_matrix.loc['input'] and index by descending order
    user_matrix = user_matrix.sort_values(by='input', axis=1, ascending=False, inplace=False)

    # find 5 highest values
    # and get the champion's name from the index
    # and print the champion's name
    idx = 0
    result_num = 0
    while result_num < 5:
        if user_matrix.loc['input'].index[idx] not in champion_list:
            result.append(user_matrix.loc['input'].index[idx])
            result_score.append(user_matrix.loc['input'].iloc[idx])
            result_num += 1
        idx += 1

    return result, result_score


champ_data = pd.read_csv('champ.csv', encoding='utf-8')

# drop rows which column '챔피언' is NaN values
champ_data = champ_data.dropna(subset=['챔피언'])

# drop rows which column '주역할군' and '부역할군' are all NaN values
champ_data = champ_data.dropna(subset=['주역할군', '부역할군'], how='all')

# Select necessary columns
features = ['주역할군', '부역할군', '탑', '미드', '바텀', '정글', '서포터']

# Fill NaN values with 0
champ_data[features] = champ_data[features].fillna(0)

# Create a feature with weighted emphasis on PrimaryRole
champ_data['combined_features'] = champ_data.apply(
    lambda row: ' '.join([str(row['주역할군'])] * 2 + [str(row['부역할군'])]) + ' ' + ' '.join(row[features[2:]].astype(str)),
    axis=1)

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(champ_data['combined_features'])


def recommend_champion(user_preferences, champ_data, tfidf_vectorizer, tfidf_matrix, input_champions):
    # Extract the main and secondary preferences
    main_preference, secondary_preference = user_preferences

    # Create text based on user preferences with emphasis on the main role
    user_text = f"{main_preference} {main_preference} {secondary_preference}"

    # Convert user text to TF-IDF
    user_tfidf = tfidf_vectorizer.transform([user_text])

    # Calculate the similarity manually
    similarity_scores = (user_tfidf * tfidf_matrix.T).A[0]

    # Sort by similarity
    similarity_scores = list(enumerate(similarity_scores))

    # Sort by similarity and exclude input champions
    similarity_scores = [champ for champ in sorted(similarity_scores, key=lambda x: x[1], reverse=True) if
                         champ_data.iloc[champ[0]]['챔피언'] not in input_champions]

    # Extract top 5 champions with scores and features
    top_champions = similarity_scores[:6]

    # Get indices of recommended champions
    recommended_indices = [idx for idx, _ in top_champions]

    # Output recommendation scores, features, and additional information
    recommended_champions = champ_data.iloc[recommended_indices][['챔피언', '주역할군', '부역할군', '탑', '미드', '바텀', '정글', '서포터']]
    recommended_champions['Recommendation_Score'] = [score for _, score in top_champions]
    recommended_champions['Similarity_Rank'] = range(1, len(top_champions) + 1)

    # Add a new '포지션' column based on all role columns
    recommended_champions['포지션'] = [
        ','.join([role for role in ['탑', '미드', '바텀', '정글', '서포터'] if champ[role] == 1])
        for _, champ in recommended_champions.iterrows()
    ]
    return recommended_champions

def get_most_common_roles(champion_list, win_rate_list, champ_data):
    roles = {'주역할군': [], '부역할군': [], '포지션': []}
    champ_win_rate_dict = dict(zip(champion_list, win_rate_list))

    # Create a list of user preferences
    champion_list = [
        champion for champion, win_rate in champ_win_rate_dict.items()
        for _ in range(int(win_rate * 100))  # Repeat champion name based on win rate (scaled by 100)
    ]

    for champion in champion_list:
        champion_info = champ_data[champ_data['챔피언'] == champion].iloc[0]
        roles['주역할군'].append(champion_info['주역할군'])
        roles['부역할군'].append(champion_info['부역할군'])
        roles['포지션'].extend([role for role in ['탑', '미드', '바텀', '정글', '서포터'] if champion_info[role] == 1])

    most_common_roles = {role: Counter(roles[role]).most_common(1)[0][0] for role in roles}
    return most_common_roles


st.title('League of Legends Champion Recommendation System')
st.header('Gachon Univ. 2023 ML 14')
name = st.text_input('plz enter your name#tag ex)Hide on bush #KR1')
if st.button('Recommend!'):
    champion_list_input, win_rate_list_input = get_user_data(name)
    print(champion_list_input, win_rate_list_input)
    if not champion_list_input:
        st.error('Unknown name. Plz check your name', icon="🚨")
    else:
        with st.spinner('Wait for it...'):
            output_user, output_score_user = predict(champion_list_input, win_rate_list_input)
            output_user_dic = []
            output_item_dic = []
            for i in range(len(output_user)):
                output_user_dic.append({output_user[i]: output_score_user[i]})
            most_common_roles = get_most_common_roles(champion_list_input, win_rate_list_input, champ_data)
            user_preferences = [most_common_roles['주역할군'], most_common_roles['포지션']]
            recommendations = recommend_champion(user_preferences, champ_data, tfidf_vectorizer, tfidf_matrix,champion_list_input)
            temp_dic = recommendations.to_dict('list')
            for i in range(len(temp_dic['챔피언'])):
                if i == 5:
                    break
                output_item_dic.append({temp_dic['챔피언'][i]: temp_dic['Recommendation_Score'][i]})

        st.success('Done!')
        st.header("Most Played Champions")
        for i in range(5):
            st.write(str(i + 1) + ': ' + champion_list_input[i])

        col1, col2 = st.columns(2)
        with col1:
            st.header("User-based")
            for i in range(5):
                st.write(str(i + 1) + ': ' + output_user[i])
            st.bar_chart(output_user_dic, use_container_width=True)

        with col2:
            st.header("Item-based")
            for i in range(5):
                st.write(str(i + 1) + ': ' + temp_dic['챔피언'][i])
            print(output_item_dic)
            st.bar_chart(output_item_dic)
