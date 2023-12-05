import numpy as np
import pandas as pd
import argparse


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
                    np.dot(user_matrix.iloc[:20][user_matrix.columns[i]], user_matrix.iloc[:20]['similarity']) / similarity_sum)

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


# 여기가 입력값입니다. 유저명으로 가져온 모스트챔, 승률 데이터만 넣으면 될 것 같아요.
champion_list_input = ['카이사', '자야', '제리', '케이틀린', '칼리스타']
win_rate_list_input = [0.60, 0.62, 0.55, 0.68, 0.55]

# predict
# insert arguments here
output, output_score = predict(champion_list_input, win_rate_list_input)

# print output
for i in range(5):
    print(output[i], output_score[i])

