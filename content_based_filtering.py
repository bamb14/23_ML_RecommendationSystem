import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

champ_data=pd.read_csv('termProject/champ.csv', encoding='utf-8')

# drop rows which column '챔피언' is NaN values
champ_data = champ_data.dropna(subset=['챔피언'])

# drop rows which column '주역할군' and '부역할군' are all NaN values
champ_data = champ_data.dropna(subset=['주역할군', '부역할군'], how='all')

# Select necessary columns
features = ['주역할군', '부역할군', '탑', '미드', '바텀', '정글', '서포터']

# Fill NaN values with 0
champ_data[features] = champ_data[features].fillna(0)

# Create a feature with weighted emphasis on PrimaryRole
champ_data['combined_features'] = champ_data.apply(lambda row: ' '.join([str(row['주역할군'])] * 2 + [str(row['부역할군'])]) + ' ' + ' '.join(row[features[2:]].astype(str)), axis=1)

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
    top_champions = similarity_scores[:10]

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

def get_most_common_roles(champion_list, win_rate_list,champ_data):
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

# Example: Input champion list and win rate list
champion_list_input = ['카이사', '자야', '제리', '케이틀린', '칼리스타']
win_rate_list_input = [0.60, 0.62, 0.55, 0.68, 0.55]

# Get most common roles from the input champions
most_common_roles = get_most_common_roles(champion_list_input, win_rate_list_input,champ_data)

# Update user_preferences with most common roles
user_preferences = [most_common_roles['주역할군'], most_common_roles['부역할군']]

# Get recommendations
recommendations = recommend_champion(user_preferences, champ_data, tfidf_vectorizer, tfidf_matrix, champion_list_input)

# Print the recommended champions with scores and features
print("Recommended champions:")
print(recommendations[['Similarity_Rank', '챔피언', '주역할군', '부역할군', '포지션', 'Recommendation_Score']].to_string(index=False))