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
    # 每处理出一个新的数据格式可以输出保存
    preparedMathData(mathGeneralDir, mathMicroDir, preparedDataDir)

def preparedMathData(mathGeneralDir, mathMicroDir, preparedDataDir):
    # math
    general01, coding01, general05, coding05, general07, coding07 = math.general_test(mathGeneralDir)


    # scoring_avg(general01, coding01, '201701数学总测得分率分布情况')
    # scoring_avg(general05, coding05, '201705数学总测得分率分布情况')
    # scoring_avg(general07, coding07, '201707数学总测得分率分布情况')
    micro1, coding1, micro2, coding2, micro3, coding3, micro4, coding4, micro5, coding5 = micro_test(mathMicroDir)

    # 选取第一次做的微测记录
    micro1 = selectEarliestMicroRecords(micro1)
    micro2 = selectEarliestMicroRecords(micro2)
    micro3 = selectEarliestMicroRecords(micro3)
    micro4 = selectEarliestMicroRecords(micro4)
    micro5 = selectEarliestMicroRecords(micro5)

    # 历次总测更新后的得分率
    mutil_general = update_concept_ability([{'test': general01, 'coding': coding01}, {'test': general05, 'coding': coding05},{'test': general07, 'coding': coding07}])
    # 历次微测更新后的得分率
    mutil_mircro = update_micro_concept_ability([{'test': micro1, 'coding': coding1, 'concept': '分式'}, {'test': micro2, 'coding': coding2, 'concept': '分式'},
                                  {'test': micro3, 'coding': coding3, 'concept': '二次根式'},{'test': micro4, 'coding': coding4, 'concept': '二次根式'},
                                  {'test': micro5, 'coding': coding5, 'concept': '变量之间的关系'}])
    # 总测微测融合
    merged_df = merge_general_micro(mutil_general, mutil_mircro)

    # 时间维度上的比较
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
    # mutil_general = update_concept_ability([{'test': general01, 'coding': coding01}, {'test': general05, 'coding': coding05}])

    # scoring_avg(general01, coding01, '201701生物总测得分率分布情况')
    # scoring_avg(general05, coding05, '201705生物总测得分率分布情况')
    # micro1, coding1, micro2, coding2, micro3, coding3, micro4, coding4, micro5, coding5, micro6, coding6, micro7, coding7, micro8, coding8, micro9, coding9, micro10, coding10 = micro_test_biology(biologyMicroDir)

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

    # mutil_mircro = update_micro_concept_ability([{'test': micro1, 'coding': coding1, 'concept': '动物的运动和行为'},
    #                                              {'test': micro2, 'coding': coding2, 'concept': '生物的起源与进化'},
    #                                              {'test': micro3, 'coding': coding3, 'concept': '生物的起源与进化'},
    #                                              {'test': micro4, 'coding': coding4, 'concept': '生物的起源与进化'},
    #                                              {'test': micro5, 'coding': coding5, 'concept': '生物的生殖和发育'},
    #                                              {'test': micro6, 'coding': coding6, 'concept': '生物的生殖和发育'},
    #                                              {'test': micro7, 'coding': coding7, 'concept': '生物的生殖和发育'},
    #                                              {'test': micro8, 'coding': coding8, 'concept': '生物的遗传和变异'},
    #                                              {'test': micro9, 'coding': coding9, 'concept': '生物的遗传和变异'},
    #                                              {'test': micro10, 'coding': coding10, 'concept': '生物的遗传和变异'}])

    # merged_df = merge_general_micro(mutil_general, mutil_mircro)

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

# 总测微测融合
def merge_general_micro(mutil_general, mutil_micro):
    # 打印微测
    # mutil_micro.to_csv(preparedDataDir + 'mutil_micro_math.csv')
    # mutil_micro_grouped = mutil_micro.groupby('核心概念').mean()
    # mutil_micro_grouped.to_csv(preparedDataDir + 'mutil_micro_grouped_math.csv')

    # 打印总测
    # mutil_general.to_csv(preparedDataDir + 'mutil_general_biology.csv')
    # mutil_general_grouped = mutil_general.groupby('核心概念').mean()
    # mutil_general_grouped.to_csv(preparedDataDir + 'mutil_general_grouped_biology.csv')

    mixed_df = mutil_general.append(mutil_micro)
    merged_df = mixed_df.groupby(['教育ID', '核心概念']).apply(cal_general_micro_mix_score)
    merged_df = merged_df[['A', 'B', 'C']]
    merged_df.reset_index(inplace=True)
    merged_general_micro = merged_df[['教育ID', '核心概念', 'A', 'B', 'C']]

    # 打印总测微测融合
    # merged_general_micro.to_csv(preparedDataDir + 'merged_general_micro_biology.csv')
    # grouped = merged_general_micro.groupby('核心概念').mean()
    # grouped.to_csv(preparedDataDir + 'merged_general_micro_grouped_biology.csv')

# 微测总测融合得分率的计算（两行，一行是总测，一行是微测）
def cal_general_micro_mix_score(frame):
    if len(frame) == 1:
        return frame

    df = frame.tail(1)

    df['A'] = frame['A'].sum() / frame['A'].count()
    df['B'] = frame['B'].sum() / frame['B'].count()
    df['C'] = frame['C'].sum() / frame['C'].count()
    return df

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

    coding1 = pd.read_excel(file_path + '2016-数学-八年级-上学期-单元微测-001（分式1）.xlsx', sheetname=2, skiprows=1, converters={'题目编码':str})
    coding2 = pd.read_excel(file_path + '2016-数学-八年级-上学期-单元微测-002（分式2）.xlsx', sheetname=2, skiprows=1, converters={'题目编码':str})
    coding3 = pd.read_excel(file_path + '2016-数学-八年级-上学期-单元微测-001（二次根式1）.xlsx', sheetname=2, skiprows=1, converters={'题目编码':str})
    coding4 = pd.read_excel(file_path + '2016-数学-八年级-上学期-单元微测-002（二次根式2）.xlsx', sheetname=2, skiprows=1, converters={'题目编码':str})
    coding5 = pd.read_excel(file_path + '2016-数学-八年级-下学期-单元微测-001（变量之间的关系）.xlsx', sheetname=2, skiprows=1, converters={'题目编码':str})

    coding1, coding2, coding5  = utility.clean_new_form_coding([coding1, coding2, coding5])
    coding3 = utility.clean_old_form_coding(coding3)
    coding4 = utility.clean_old_form_coding(coding4)

    return df1, coding1, df2, coding2, df3, coding3, df4, coding4, df5, coding5

def adjust_micro_q_code(coding):
    copy_df = coding.copy()
    copy_df['题目编码'] = copy_df['题目编码'].apply(lambda x: '0' + str(x))
    return copy_df

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
    # nan to zero
    df.fillna(0, inplace=True)
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

# 原始数据和编码表数据合并，以编码表为基表
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
    return stu_concept_ability #.loc[(stu_concept_ability['教育ID'] == '09067195') | (stu_concept_ability['教育ID'] == '09067322')]

# 某核心概念一能力要素的历次得分率弄成一行，利用update_score迭代得到最新的得分率
def update_concept_ability_score(df):
    grouped = df.groupby(['教育ID', '核心概念', '学习表现指标代码'])
    grouped = grouped.apply(lambda x: update_score(x, 1, 2, x.iloc[:, 3:].values.flatten().tolist(), 1))
    grouped = grouped[['教育ID', '核心概念', '学习表现指标代码', 'final_score']]
    return grouped

# scores为某能力历史分的数组
def update_score(x, weight1, weight2, scores, axis):
    for i, score in enumerate(scores):
        if i == 0:
            final_score = score
            continue

        if (pd.isnull(final_score)) & (pd.isnull(score)):
            continue

        w1 = weight1
        w2 = weight2

        if pd.isnull(final_score):
            final_score = 0
            w1 = 0
        if pd.isnull(score):
            score = 0
            w2 = 0

        final_score = (final_score * w1 + score * w2) / (w1 + w2)

    if axis == 1:
        x['final_score'] = final_score
        return x
    elif axis == 0:
        return final_score

# 历次总测更新获得各核心概念能力得分率
def update_concept_ability(test_list):
    tests_formated = []
    for obj in test_list:
        test = obj['test']
        coding = obj['coding']
        tests_formated.append(get_stu_concept_ability(test, coding))
    for i, general in enumerate(tests_formated):
        if i == 0:
            df = general
        else:
            df = pd.merge(df, general, on=['教育ID', '核心概念', '学习表现指标代码'], how='outer')
        # print(df)
    updated_df = update_concept_ability_score(df)
    updated_df_formated = pd.DataFrame()
    i = 0
    for group, frame in updated_df.groupby(['教育ID', '核心概念']):
        formated_frame = pd.pivot_table(frame, index=['教育ID', '核心概念'], columns=['学习表现指标代码'], values='final_score')
        if i == 0:
            updated_df_formated = formated_frame
        else:
            updated_df_formated = updated_df_formated.append(formated_frame)
        i = i + 1
        #print(type())
    updated_df_formated.reset_index(inplace=True)
    return updated_df_formated

# 历次微测更新获得各核心概念能力得分率
def update_micro_concept_ability(test_list):
    tests_formated = []
    for i, obj in enumerate(test_list):
        test = obj['test']
        coding = obj['coding']
        concept = obj['concept']
        # 计算每道题的平均得分率不需要“总分”这一列
        test.drop('总分', axis=1, inplace=True)
        # 计算每道题的平均得分率
        avg_score_test = utility.avg_score(test, coding)
        # 暂时不需要索引
        avg_score_test.reset_index(inplace=True)
        # 给微测编码表添加核心概念
        # coding['核心概念'] = concept
        coding = shorten_ability(coding)
        # 将题目编码转化成学习表现指标代码
        avg_score_test.columns.values[2:] = list(map(lambda x: coding.loc[x]['学习表现指标代码'], avg_score_test.columns.values[2:]))
        avg_score_test['核心概念'] = concept
        avg_score_test.set_index(['教育ID', '作答日期', '核心概念'], inplace=True)
        # 计算三个能力的平均得分率
        avg_score_test = avg_score_test.groupby(avg_score_test.columns, axis=1).mean()
        tests_formated.append(avg_score_test)

    # 连接所有的微测
    df = pd.concat(tests_formated)
    df.reset_index(inplace=True)
    # print(df[df['教育ID'] == '09067322'])
    # 同一天做多套微测的学生，其A,B,C能力用均值表示
    df = df.groupby(['教育ID', '作答日期', '核心概念']).mean()
    df.reset_index(inplace=True)
    df.sort_values(['教育ID', '作答日期'], inplace=True)
    # print(df[df['教育ID'] == '09067322'])
    # df.to_csv(preparedDataDir + 'examine groupby data.csv')
    #df = df[(df['教育ID'] == '09067322') | (df['教育ID'] == '09067195')]
    df.drop('作答日期', axis=1, inplace=True)
    # 计算历次
    updated_df = df.groupby(['教育ID', '核心概念']).apply(cal_concept_ability)
    updated_df = updated_df.loc[:, 'A':'C']
    updated_df.reset_index(inplace=True)
    updated_df = updated_df[['教育ID', '核心概念', 'A', 'B', 'C']]
    return updated_df

# 获取编码表里的核心概念和学习表现指标代码
def shorten_ability(coding):
    updated_coding = coding.copy()
    updated_coding['学习表现指标代码'] = coding['学习表现指标代码'].apply(lambda x: x[0])
    return updated_coding

# 将题目编码转换成能力
def replace_q_code_with_concept(general, coding):
    copy_df = general.copy()
    copy_df.columns = list(map(lambda x: coding.loc[x]['学习表现指标代码'], general.columns))
    return copy_df

# 将能力的列组成数组传到update_score方法计算迭代后的能力得分率
def cal_concept_ability(stu_time_concept_ability_df):
    if len(stu_time_concept_ability_df) == 1:
        return stu_time_concept_ability_df
    df = stu_time_concept_ability_df.tail(1)
    df['A'] = update_score(None, 1, 2, stu_time_concept_ability_df['A'].tolist(), axis=0)
    df['B'] = update_score(None, 1, 2, stu_time_concept_ability_df['B'].tolist(), axis=0)
    df['C'] = update_score(None, 1, 2, stu_time_concept_ability_df['C'].tolist(), axis=0)
    return df

if __name__ == "__main__": main()