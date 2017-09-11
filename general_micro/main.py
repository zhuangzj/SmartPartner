import pandas as pd
from scipy.stats import norm
import SmartPartner.code.general_micro.math as math
import SmartPartner.code.general_micro.biology as biology
import SmartPartner.code.general_micro.utility as utility
import matplotlib.pyplot as plt
from matplotlib import font_manager
chinesefont = font_manager.FontProperties()
chinesefont.set_family('SimHei')

mathGeneralDir = 'D:/PycharmProjects/SmartPartner/data/general_test/math/'
mathMicroDir = 'D:/PycharmProjects/SmartPartner/data/micro_test/math/'
biologyGeneralDir = 'D:/PycharmProjects/SmartPartner/data/general_test/biology/'
biologyMicroDir = 'D:/PycharmProjects/SmartPartner/data/micro_test/biology/'
outputDir = 'D:/PycharmProjects/SmartPartner/output/'
preparedDataDir = 'D:/PycharmProjects/SmartPartner/preparedData/'

def main():
    preparedMathData(mathGeneralDir, mathMicroDir, preparedDataDir)

def preparedMathData(mathGeneralDir, mathMicroDir, preparedDataDir):
    # math
    general01, coding01, general05, coding05, general07, coding07 = math.general_test(mathGeneralDir)
    general_first = get_stu_concept_ability(general01, coding01)
    general_second = get_stu_concept_ability(general05, coding05)
    general_third = get_stu_concept_ability(general07, coding07)
    for i, general in enumerate([general_first, general_second, general_third]):
        if i == 0:
            df = general
        else:
            df = pd.merge(df, general, on=['教育ID', '核心概念', '学习表现指标代码'], how='outer')
        # print(df)
    updated_df = update_concept_ability_score(df)
    print(pd.pivot_table(updated_df, index=['核心概念', '学习表现指标代码'], columns=['教育ID'], values='final_score'))
    # concept_ability(general01, coding01)
    # scoring_avg(general01, coding01, '201701数学总测得分率分布情况')
    # scoring_avg(general05, coding05, '201705数学总测得分率分布情况')
    # scoring_avg(general07, coding07, '201707数学总测得分率分布情况')
    # micro1, coding1, micro2, coding2, micro3, coding3, micro4, coding4, micro5, coding5 = micro_test(mathMicroDir)
    #
    # micro1 = selectEarliestMicroRecords(micro1)
    # micro2 = selectEarliestMicroRecords(micro2)
    # micro3 = selectEarliestMicroRecords(micro3)
    # micro4 = selectEarliestMicroRecords(micro4)
    # micro5 = selectEarliestMicroRecords(micro5)

    # general01_fraction_total_score_avg = cal_concept_total_score_avg_scoing(general01, coding01, '分式')
    # general_score_comparison(general01_fraction_total_score_avg, '201701', [micro1, micro2], '分式', '2017-01-14',
    #                                     '2017-01-16')
    # general01_quadratic_total_score_avg = cal_concept_total_score_avg_scoing(general01, coding01, '二次根式')
    # general_score_comparison(general01_quadratic_total_score_avg, '201701', [micro3, micro4], '二次根式', '2017-01-14',
    #                          '2017-01-16')
    # general05_quadratic_total_score_avg = cal_concept_total_score_avg_scoing(general05, coding05, '二次根式')
    # general_score_comparison(general05_quadratic_total_score_avg, '201705', [micro3, micro4], '二次根式', '2017-05-02',
    #                                     '2017-05-02')
    # general07_varcorr_total_score_avg = cal_concept_total_score_avg_scoing(general07, coding07, '变量之间的关系')
    # general_score_comparison(general07_varcorr_total_score_avg, '201707', [micro5], '变量之间的关系', '2017-07-05', '2017-07-07')

    # general01, coding01, general05, coding05 = biology.general_test(biologyGeneralDir)
    # scoring_avg(general01, coding01, '201701生物总测得分率分布情况')
    # scoring_avg(general05, coding05, '201705生物总测得分率分布情况')
    # micro1, coding1, micro2, coding2, micro3, coding3, micro4, coding4, micro5, coding5, micro6, coding6, micro7, coding7, micro8, coding8, micro9, coding9, micro10, coding10 = micro_test_biology(biologyMicroDir)
    #
    # micro1 = selectEarliestMicroRecords(micro1)
    # micro2 = selectEarliestMicroRecords(micro2)
    # micro3 = selectEarliestMicroRecords(micro3)
    # micro4 = selectEarliestMicroRecords(micro4)
    # micro5 = selectEarliestMicroRecords(micro5)
    # micro6 = selectEarliestMicroRecords(micro6)
    # micro7 = selectEarliestMicroRecords(micro7)
    # micro8 = selectEarliestMicroRecords(micro8)
    # micro9 = selectEarliestMicroRecords(micro9)
    # micro10 = selectEarliestMicroRecords(micro10)

    # 时间维度上的比较
    # general01_behavior_total_score_avg = cal_concept_total_score_avg_scoing(general01, coding01, '动物的运动和行为')
    # general_score_comparison(general01_behavior_total_score_avg, '201701', [micro1], '动物的运动和行为', '2017-01-14', '2017-01-16')
    # general01_origin_total_score_avg = cal_concept_total_score_avg_scoing(general01, coding01, '生物的起源与进化')
    # general_score_comparison(general01_origin_total_score_avg, '201701', [micro2, micro3, micro4], '生物的起源与进化', '2017-01-14', '2017-01-16')
    # general01_reproduction_total_score_avg = cal_concept_total_score_avg_scoing(general01, coding01, '生物的生殖和发育')
    # general_score_comparison(general01_reproduction_total_score_avg, '201701', [micro5, micro6, micro7], '生物的生殖和发育', '2017-01-14', '2017-01-16')
    # general01_inheritance_total_score_avg = cal_concept_total_score_avg_scoing(general01, coding01, '生物的遗传和变异')
    # general_score_comparison(general01_inheritance_total_score_avg, '201701', [micro8, micro9, micro10], '生物的遗传和变异', '2017-01-14', '2017-01-16')

    # 得分率 （没有算时间）
    # micro1_total_score_avg = cal_microtest_total_score_scoing_avg_scoing(micro1, coding1)
    # micro2_total_score_avg = cal_microtest_total_score_scoing_avg_scoing(micro2, coding2)
    # micro3_total_score_avg = cal_microtest_total_score_scoing_avg_scoing(micro3, coding3)
    # micro4_total_score_avg = cal_microtest_total_score_scoing_avg_scoing(micro4, coding4)
    # micro5_total_score_avg = cal_microtest_total_score_scoing_avg_scoing(micro5, coding5)
    # micro6_total_score_avg = cal_microtest_total_score_scoing_avg_scoing(micro6, coding6)
    # micro7_total_score_avg = cal_microtest_total_score_scoing_avg_scoing(micro7, coding7)
    # micro8_total_score_avg = cal_microtest_total_score_scoing_avg_scoing(micro8, coding8)
    # micro9_total_score_avg = cal_microtest_total_score_scoing_avg_scoing(micro9, coding9)
    # micro10_total_score_avg = cal_microtest_total_score_scoing_avg_scoing(micro10, coding10)

    # scoring([general01_behavior_total_score_avg, micro1_total_score_avg], '动物的运动和行为')
    # scoring([general01_origin_total_score_avg, micro2_total_score_avg, micro3_total_score_avg, micro4_total_score_avg], '生物的起源与进化')



# 筛选有用数据（数学学科）
def get_useful_data(col_start, col_end, df):
    df = df.iloc[2:, col_start:col_end]
    df.drop('题目编码', axis=1, inplace=True)
    df.drop('姓名', axis=1, inplace=True)
    df.set_index(['教育ID'], inplace=True)
    return df

def micro_test(file_path):
    # read data
    df1 = pd.read_excel(file_path + 'with-date/2016-数学-八年级-上学期-单元微测-001（分式1）-date.xlsx', converters={'学号':str, '作答日期':pd.to_datetime})
    df2 = pd.read_excel(file_path + 'with-date/2016-数学-八年级-上学期-单元微测-002（分式2）-date.xlsx', converters={'学号':str, '作答日期':pd.to_datetime})
    df3 = pd.read_excel(file_path + 'with-date/2016-数学-八年级-上学期-单元微测-001（二次根式1）-date.xlsx', converters={'学号':str, '作答日期':pd.to_datetime})
    df4 = pd.read_excel(file_path + 'with-date/2016-数学-八年级-上学期-单元微测-002（二次根式2）-date.xlsx', converters={'学号':str, '作答日期':pd.to_datetime})
    df5 = pd.read_excel(file_path + 'with-date/2016-数学-八年级-下学期-单元微测-001（变量之间的关系）-date.xlsx', converters={'学号':str, '作答日期':pd.to_datetime})

    df1 = clean_micro_test(df1, 'normal')
    df2 = clean_micro_test(df2, 'normal')
    df3 = clean_micro_test(df3, 'normal')
    df4 = clean_micro_test(df4, 'abnormal')
    df5 = clean_micro_test(df5, 'normal')

    coding1 = pd.read_excel(file_path + '2016-数学-八年级-上学期-单元微测-001（分式1）.xlsx', sheetname=2, skiprows=1)
    coding2 = pd.read_excel(file_path + '2016-数学-八年级-上学期-单元微测-002（分式2）.xlsx', sheetname=2, skiprows=1)
    coding3 = pd.read_excel(file_path + '2016-数学-八年级-上学期-单元微测-001（二次根式1）.xlsx', sheetname=2, skiprows=1)
    coding4 = pd.read_excel(file_path + '2016-数学-八年级-上学期-单元微测-002（二次根式2）.xlsx', sheetname=2, skiprows=1)
    coding5 = pd.read_excel(file_path + '2016-数学-八年级-下学期-单元微测-001（变量之间的关系）.xlsx', sheetname=2, skiprows=1)

    coding1, coding2, coding5  = utility.clean_new_form_coding([coding1, coding2, coding5])
    coding3 = utility.clean_old_form_coding(coding3)
    coding4 = utility.clean_old_form_coding(coding4)
    return df1, coding1, df2, coding2, df3, coding3, df4, coding4, df5, coding5

def read_micro_test(file_path, q_code_subtract_way):
    df = pd.read_excel(file_path, converters={'学号':str, '作答日期':pd.to_datetime})
    df = clean_micro_test(df, q_code_subtract_way)
    return df

def micro_test_biology(file_path):
    df1 = read_micro_test(file_path + '/with-date/2016-生物-八年级-上学期-单元微测-动物的运动和行为T001-date.xlsx', 'normal')
    df2 = read_micro_test(file_path + '/with-date/2016-生物-八年级-上学期-单元微测-生命起源与生物进化T001-date.xlsx', 'normal')
    df3 = read_micro_test(file_path + '/with-date/2016-生物-八年级-上学期-单元微测-生命起源与生物进化T002-date.xlsx', 'normal')
    df4 = read_micro_test(file_path + '/with-date/2016-生物-八年级-上学期-单元微测-生命起源与生物进化T003-date.xlsx', 'normal')
    df5 = read_micro_test(file_path + '/with-date/2016-生物-八年级-上学期-单元微测-生物的生殖与发育T001-date.xlsx', 'normal')
    df6 = read_micro_test(file_path + '/with-date/2016-生物-八年级-上学期-单元微测-生物的生殖与发育T002-date.xlsx', 'normal')
    df7 = read_micro_test(file_path + '/with-date/2016-生物-八年级-上学期-单元微测-生物的生殖与发育T003-date.xlsx', 'normal')
    df8 = read_micro_test(file_path + '/with-date/2016-生物-八年级-上学期-单元微测-遗传与变异T001-date.xlsx', 'normal')
    df9 = read_micro_test(file_path + '/with-date/2016-生物-八年级-上学期-单元微测-遗传与变异T002-date.xlsx', 'normal')
    df10 = read_micro_test(file_path + '/with-date/2016-生物-八年级-上学期-单元微测-遗传与变异T003-date.xlsx', 'normal')

    coding1 = pd.read_excel(file_path + '北京市2016-生物-八年级-上学期-单元微测-动物的运动和行为T001.xlsx', sheetname=2, skiprows=1)
    coding2 = pd.read_excel(file_path + '北京市2016-生物-八年级-上学期-单元微测-生命起源与生物进化T001.xlsx', sheetname=2, skiprows=1)
    coding3 = pd.read_excel(file_path + '北京市2016-生物-八年级-上学期-单元微测-生命起源与生物进化T002.xlsx', sheetname=2, skiprows=1)
    coding4 = pd.read_excel(file_path + '北京市2016-生物-八年级-上学期-单元微测-生命起源与生物进化T003.xlsx', sheetname=2, skiprows=1)
    coding5 = pd.read_excel(file_path + '北京市2016-生物-八年级-上学期-单元微测-生物的生殖与发育T001.xlsx', sheetname=2, skiprows=1)
    coding6 = pd.read_excel(file_path + '北京市2016-生物-八年级-上学期-单元微测-生物的生殖与发育T002.xlsx', sheetname=2, skiprows=1)
    coding7 = pd.read_excel(file_path + '北京市2016-生物-八年级-上学期-单元微测-生物的生殖与发育T003.xlsx', sheetname=2, skiprows=1)
    coding8 = pd.read_excel(file_path + '北京市2016-生物-八年级-上学期-单元微测-遗传与变异T001.xlsx', sheetname=2, skiprows=1)
    coding9 = pd.read_excel(file_path + '北京市2016-生物-八年级-上学期-单元微测-遗传与变异T002.xlsx', sheetname=2, skiprows=1)
    coding10 = pd.read_excel(file_path + '北京市2016-生物-八年级-上学期-单元微测-遗传与变异T003.xlsx', sheetname=2, skiprows=1)

    coding1, coding2, coding3, coding4, coding5, coding6, coding7, coding8, coding9, coding10 = utility.clean_new_form_coding(
        [coding1, coding2, coding3, coding4, coding5, coding6, coding7, coding8, coding9, coding10])

    return df1, coding1, df2, coding2, df3, coding3, df4, coding4, df5, coding5, df6, coding6, df7, coding7, df8, coding8, df9, coding9, df10, coding10

def clean_micro_test(df, q_code_subtract_way):
    df = df.iloc[:, 6:]
    df.drop('性别', axis=1, inplace=True)
    df.rename(columns={'学号': '教育ID'}, inplace=True)
    df.drop('姓名', axis=1, inplace=True)
    df.columns.values[2] = '总分'
    df['总分'] = df['总分'].astype(int)
    # 对应编码
    if q_code_subtract_way == 'normal':
        df.columns.values[3:] = list(map(lambda x: x[1:-2], df.columns[3:]))
    else: # 有的微测如micro4编码表对应的编码不是P之后的六位数值，而是P之后的四位数值加上后两位数值
        df.columns.values[3:] = list(map(lambda x: x[1:5] + x[-2:], df.columns[3:]))
    return df


# group -> sort -> drop duplicate(first) -> merge
# sort two cols -> group -> fist record
def selectEarliestMicroRecords(df_duplicate):
    # df = df.groupby(df.index).apply(pd.DataFrame.sort_values,'作答日期')
    df = df_duplicate.copy()
    df.sort_values(['教育ID', '作答日期'], inplace=True)
    df = df.groupby('教育ID').first()
    return df

# 计算总测中一个核心概念的总分得分率(concept='' -> 该试卷的所有核心概念的总分得分率)
def cal_concept_total_score_avg_scoing(test, coding, concept = '总分'):
    if concept == '总分':
        series = test.sum(axis=1)
        series.rename('总分', inplace=True)
        total_score = utility.get_total_score_from_coding(coding)
    else :
        test_concept = utility.replace_q_code_with_concept(test, coding)
        series = utility.get_concept_total_score(test_concept, concept)
        total_score = utility.get_total_score_from_coding(coding, concept)
        total_score = total_score[concept]

    df = series.to_frame()
    df['总分得分率'] = df[concept].apply(lambda x: x/ total_score)
    #series_concept.to_csv(preparedDataDir + 'MathMicro01ConceptTotalScores.csv')
    return df['总分得分率']

def cal_microtest_total_score_scoing_avg_scoing(micro, coding):
    test_total_score = coding['总分'].sum()
    series = micro['总分'].copy()
    series = series.map(lambda x: x/test_total_score)
    return series

# 考总测前参加微测的学生在总测的得分和没参加微测的学生在总测的得分
# 输入：
def general_score_comparison(general, general_name, micro_list, concept, start_time, end_time):
    for i, micro in enumerate(micro_list):
        avg_micro_before, avg_micro_after = math.get_micro_time_groups(micro, start_time, end_time, concept, i+1)
        if i == 0:
            df = pd.merge(general.to_frame(), avg_micro_before, left_index=True, right_index=True,
                          how='left')
        else:
            df = pd.merge(df, avg_micro_before, left_index=True, right_index=True, how='left')

    df.to_csv(preparedDataDir + general_name + '总测前参加微测的学生在总测的表现--' + concept + '.csv')

def scoring(test_list, concept):
    df = pd.concat(test_list, join='inner', axis=1)
    df.to_csv(preparedDataDir + '同一核心概念的测试的总分得分率--' + concept + '.csv')

# 每题的得分率的平均
def cal_avg_scoring(general, coding):
    df = utility.avg_score(general, coding)
    df['平均得分率'] = df.mean(axis=1)
    return df['平均得分率']

# scoing -> bin into diff category (0, 0.1, 0.2, ..., 0.9, 1) -> count freq
def scoring_avg(general, coding, general_name):
    series_total_scoring = cal_concept_total_score_avg_scoing(general, coding)
    series_avg_scoring = cal_avg_scoring(general, coding)
    total_mu, total_std = norm.fit(series_total_scoring)
    print('total:')
    print('mu:' + str(total_mu) + ', ' + 'std:' + str(total_std))
    avg_mu, avg_std = norm.fit(series_avg_scoring)
    print('avg:')
    print('mu:' + str(avg_mu) + ', ' + 'std:' + str(avg_std))
    df_scoring = pd.concat([series_total_scoring, series_avg_scoring], axis=1)
    # df_scoring = bin_scoring(df_scoring)
    # freq_total_scoring = df_scoring['总分得分率binned'].value_counts().sort_index()
    # # freq_total_scoring.plot(kind='line', label="总分得分率")
    # freq_avg_scoring = df_scoring['平均得分率binned'].value_counts().sort_index()
    # # freq_avg_scoring.plot(kind='line', label="平均得分率")
    # freq = pd.concat([freq_total_scoring, freq_avg_scoring], axis=1)
    # freq.rename(columns={'总分得分率binned': '总分得分率', '平均得分率binned': '平均得分率'}, inplace=True)
    # freq.plot(kind='line')
    # plt.legend(prop=chinesefont)
    # plt.title(general_name, fontproperties=chinesefont)
    # plt.show()
    # series_total_scoring.to_csv(preparedDataDir + general_name + '--总分得分率.csv')
    # series_avg_scoring.to_csv(preparedDataDir + general_name + '--平均得分率.csv')
    df_scoring.to_csv(preparedDataDir + general_name + '.csv')

def bin_scoring(df_scoring):
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    df_scoring['总分得分率binned'] = pd.cut(df_scoring['总分得分率'], bins)
    df_scoring['平均得分率binned'] = pd.cut(df_scoring['平均得分率'], bins)
    return df_scoring

def concept_ability(test, code):
    # print(test.head())
    # test.columns = pd.MultiIndex.from_product([test.columns, ['C']])
    df = utility.avg_score(test, code)
    code_df = code[['核心概念', '学习表现指标代码']]
    copy_code_df = code_df.copy()
    copy_code_df['学习表现指标代码'] = code_df['学习表现指标代码'].apply(lambda x: x[0])
    data = []
    for i, row in df.iterrows():
        stu_score = pd.concat([row, copy_code_df], axis=1)
        stu_score.reset_index(inplace=True)
        pt = pd.pivot_table(stu_score,columns=['核心概念', '学习表现指标代码'])
        for row in pt.iteritems():
            data.append({'教育ID': row[0][0], '核心概念': row[0][1], '学习表现指标代码': row[0][2], '得分率': row[1]})
    stu_concept_ability = pd.DataFrame(data)
    print(stu_concept_ability)

def get_stu_concept_ability(test, coding):
    df = utility.avg_score(test, coding)
    code_df = coding[['核心概念', '学习表现指标代码']]
    copy_code_df = code_df.copy()
    copy_code_df['学习表现指标代码'] = code_df['学习表现指标代码'].apply(lambda x: x[0])
    data = []
    for i, row in df.iterrows():
        stu_score = pd.concat([row, copy_code_df], axis=1)
        stu_score.reset_index(inplace=True)
        pt = pd.pivot_table(stu_score, columns=['核心概念', '学习表现指标代码'])
        for row in pt.iteritems():
            data.append({'教育ID': row[0][0], '核心概念': row[0][1], '学习表现指标代码': row[0][2], '得分率': row[1]})
    stu_concept_ability = pd.DataFrame(data)
    stu_concept_ability = stu_concept_ability[['教育ID', '核心概念', '学习表现指标代码', '得分率']]
    return stu_concept_ability.loc[(stu_concept_ability['教育ID'] == '07073703') | (stu_concept_ability['教育ID'] == '09071100')]
    # print(stu_concept_ability.loc[stu_concept_ability['教育ID'] == '07073703'])

def update_concept_ability_score(df):
    grouped = df.groupby(['教育ID', '核心概念', '学习表现指标代码'])
    grouped = grouped.apply(lambda x: update_score(x, 1, 2))
    grouped = grouped[['教育ID', '核心概念', '学习表现指标代码', 'final_score']]
    print(grouped)
    return grouped

def update_score(x, weight1, weight2):
    scores = x.iloc[:, 3:].values.flatten().tolist()
    for i, score in enumerate(scores):
        if i == 0:
            final_score = score
            continue

        w1 = weight1
        w2 = weight2
        if pd.isnull(final_score):
            final_score = 0
            w1 = 0
        if pd.isnull(score):
            score = 0
            w2 = 0
        if (w1 == 0) & (w2 == 0):
            continue

        final_score = (final_score * w1 + score * w2) / (w1 + w2)

    x['final_score'] = final_score
    return x


if __name__ == "__main__": main()