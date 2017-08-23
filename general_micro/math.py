import pandas as pd
import SmartPartner.code.general_micro.utility as utility
from scipy.stats import pearsonr
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib import font_manager
chinesefont = font_manager.FontProperties()
chinesefont.set_family('SimHei')

def main():
    general01, coding01, general05, coding05, general07, coding07 = general_test('D:/PycharmProjects/SmartPartner/data/general_test/math/')
    micro1, coding1, micro2, coding2, micro3, coding3, micro4, coding4, micro5, coding5 = micro_test('D:/PycharmProjects/SmartPartner/data/micro_test/math/')

    # 平均得分率
    avg_general01 = utility.avg_score(general01, coding01)
    avg_general05 = utility.avg_score(general05, coding05)
    avg_general07 = utility.avg_score(general07, coding07)
    avg_micro1 = utility.avg_score(micro1, coding1)
    avg_micro2 = utility.avg_score(micro2, coding2)
    avg_micro3 = utility.avg_score(micro3, coding3)
    avg_micro4 = utility.avg_score(micro4, coding4)
    avg_micro5 = utility.avg_score(micro5, coding5)

    # 难度分析
    concept_avg_general01 = replace_q_code_with_concept(avg_general01, coding01)
    concept_avg_general05 = replace_q_code_with_concept(avg_general05, coding05)
    concept_avg_general07 = replace_q_code_with_concept(avg_general07, coding07)
        # 分式的测试难度
    fraction_difficulty = tests_difficulty([{'name':'总测-201701', 'content': concept_avg_general01, 'type': 'general'},
                                            {'name':'微测-分式1', 'content': avg_micro1, 'type': 'micro'},
                                            {'name': '微测-分式2', 'content': avg_micro2, 'type': 'micro'}], '分式')
        # 二次根式的测试难度
    quadratic_difficulty = tests_difficulty([{'name':'总测-201701', 'content': concept_avg_general01, 'type': 'general'},
                                             {'name': '总测-201705', 'content': concept_avg_general05, 'type': 'general'},
                                             {'name':'微测-二次根式1', 'content': avg_micro3, 'type': 'micro'},
                                             {'name': '微测-二次根式2', 'content': avg_micro4, 'type': 'micro'}], '二次根式')
        # 变量之间的关系的测试难度
    vars_corr_difficulty = tests_difficulty([{'name':'总测-201707', 'content': concept_avg_general07, 'type': 'general'},
                                             {'name':'微测-变量之间的关系', 'content': avg_micro5, 'type': 'micro'}], '变量之间的关系')

    # 相关性分析
        # 测试总分来做相关
    # concept_general01 = replace_q_code_with_concept(general01, coding01)
    # concept_general05 = replace_q_code_with_concept(general05, coding05)
    # concept_general07 = replace_q_code_with_concept(general07, coding07)
    #         # 201701在分式概念的总分和分式微测1的总分做相关性分析
    # corr_gener01_mic1, p_val_gener01_mic1 = corr_analysis(concept_general01, '分式', micro1, 'general-micro')
    #         # 201701在分式概念的总分和分式微测2的总分做相关性分析
    # corr_gener01_mic2, p_val_gener01_mic2 = corr_analysis(concept_general01, '分式', micro2, 'general-micro')
    #         # 201701在二次根式概念的总分和二次根式微测1的总分做相关性分析
    # corr_gener01_mic3, p_val_gener01_mic3 = corr_analysis(concept_general01, '二次根式', micro3, 'general-micro')
    #         # 201701在二次根式概念的总分和二次根式微测2的总分做相关性分析
    # corr_gener01_mic4, p_val_gener01_mic4 = corr_analysis(concept_general01, '二次根式', micro4, 'general-micro')
    #         # 201705在二次根式概念的总分和二次根式微测1的总分做相关性分析
    # corr_gener05_mic3, p_val_gener05_mic3 = corr_analysis(concept_general05, '二次根式', micro3, 'general-micro')
    #         # 201705在二次根式概念的总分和二次根式微测2的总分做相关性分析
    # corr_gener05_mic4, p_val_gener05_mic4 = corr_analysis(concept_general05, '二次根式', micro4, 'general-micro')
    #         # 201705在二次根式概念的总分和二次根式微测2的总分做相关性分析
    # corr_gener07_mic5, p_val_gener07_mic5 = corr_analysis(concept_general07, '变量之间的关系', micro5, 'general-micro')
        # 微测和总测同一概念的同一能力水平做相关性分析
        # 微测与微测间的相关性分析；等级分；斯皮尔曼分析
    # corr_mic1_mic2, p_val_mic1_mic2 = corr_analysis(micro1, '分式', micro2, 'micro-micro')
    # corr_mic3_mic4, p_val_mic3_mic4 = corr_analysis(micro3, '二次根式', micro4, 'micro-micro')


def general_test(file_path):
    # read data
    df01, coding01 = read_data_from_general_test(file_path + '201701-初二数学总测.xls', file_path + '201701-初二数学编码.xlsx')
    df05, coding05 = read_data_from_general_test(file_path + '201705-初二数学总测.xls', file_path + '201705-初二数学编码.xlsx')
    df07, coding07 = read_data_from_general_test(file_path + '201707-初二数学总测.xls', file_path + '201707-初二数学编码.xlsx')

    # clean data
        # select test useful columns
    df01 = clean_general_test(0, 31, df01)
    df05 = clean_general_test(0, 29, df05)
    df07 = clean_general_test(0, 38, df07)
        # 将201701最后三个编码从数字转成文本
    df01.columns.values[-3:] = list(map(lambda x: '0' + str(x), df01.columns[-3:]))
        # get coding useful data
    coding01 = utility.clean_new_form_coding(coding01)
    coding05 = utility.clean_new_form_coding(coding05)
    coding07 = utility.clean_new_form_coding(coding07)

    return df01, coding01, df05, coding05, df07, coding07

# 读试卷成绩和编码
def read_data_from_general_test(test_file, coding_file):
    df = pd.read_excel(test_file, converters={'教育ID':str})
    coding = pd.read_excel(coding_file, converters={'题目编码':str}, skiprows=1)
    return df, coding

# 试卷有效cols选择
def clean_general_test(col_start, col_end, df):
    df = df.iloc[2:, col_start:col_end]
    df.drop('题目编码', axis=1, inplace=True)
    df.set_index(['教育ID', '姓名'], inplace=True)
    return df

def micro_test(file_path):
    # read data
    df1, coding1 = read_data_from_micro_test(file_path + '2016-数学-八年级-上学期-单元微测-001（分式1）.xlsx')
    df2, coding2 = read_data_from_micro_test(file_path + '2016-数学-八年级-上学期-单元微测-002（分式2）.xlsx')
    df3, coding3 = read_data_from_micro_test(file_path + '2016-数学-八年级-上学期-单元微测-001（二次根式1）.xlsx')
    df4, coding4 = read_data_from_micro_test(file_path + '2016-数学-八年级-上学期-单元微测-002（二次根式2）.xlsx')
    df5, coding5 = read_data_from_micro_test(file_path + '2016-数学-八年级-下学期-单元微测-001（变量之间的关系）.xlsx')

    # clean data
        # add data & tidy data
    df1 = clean_micro_test(df1, file_path + '/with-date/2016-数学-八年级-上学期-单元微测-001（分式1）-date.xlsx', 'normal')
    df2 = clean_micro_test(df2, file_path + '/with-date/2016-数学-八年级-上学期-单元微测-002（分式2）-date.xlsx', 'normal')
    df3 = clean_micro_test(df3, file_path + '/with-date/2016-数学-八年级-上学期-单元微测-001（二次根式1）-date.xlsx', 'normal')
    df4 = clean_micro_test(df4, file_path + '/with-date/2016-数学-八年级-上学期-单元微测-002（二次根式2）-date.xlsx', 'abnormal')
    df5 = clean_micro_test(df5, file_path + '/with-date/2016-数学-八年级-下学期-单元微测-001（变量之间的关系）-date.xlsx', 'normal')

    coding1 = utility.clean_new_form_coding(coding1)
    coding2 = utility.clean_new_form_coding(coding2)
    coding3 = utility.clean_old_form_coding(coding3)
    coding4 = utility.clean_old_form_coding(coding4)
    coding5 = utility.clean_new_form_coding(coding5)

    return df1, coding1, df2, coding2, df3, coding3, df4, coding4, df5, coding5

# 读微测数据
def read_data_from_micro_test(filename):
    df = pd.read_excel(filename, sheetname=1, converters={'学号': str})
    coding = pd.read_excel(filename, sheetname=2, skiprows=1)
    return df, coding

# 清洗调整微测数据
def clean_micro_test(df, filename_date, q_code_subtract_way):
    # add test date
    df = utility.add_date(df, filename_date)
    # tidy data
    df = df.iloc[:, 6:]
    df.drop('性别', axis=1, inplace=True)
    df.rename(columns={'学号': '教育ID'}, inplace=True)
    df.set_index(['教育ID', '姓名'], inplace=True)
    if q_code_subtract_way == 'normal':
        df.columns.values[:-1] = list(map(lambda x: x[1:-2], df.columns[:-1]))
    else: # 有的微测如micro4编码表对应的编码不是P之后的六位数值，而是P之后的四位数值加上后两位数值
        df.columns.values[:-1] = list(map(lambda x: x[1:5] + x[-2:], df.columns[:-1]))
    return df

# 测试成绩的相关性分析
def corr_analysis(df_x, concept, df_y, objs):
    if objs == 'general-micro':
        series_x = get_concept_total_score(df_x, concept)
        series_y = get_total_score(df_y, 0, len(df_y.columns)-1)
    elif objs == 'micro-micro':
        series_x = get_total_score(df_x, 0, len(df_x.columns) - 1)
        series_y = get_total_score(df_y, 0, len(df_y.columns) - 1)
    df = pd.merge(series_x.to_frame(), series_y.to_frame(), left_index=True, right_index=True) # 两测试都参加的学生
    x = df.iloc[:, 0]
    y = df.iloc[:, 1]
    corr, p_value = pearsonr(x, y)
    return corr, p_value

# 计算总分（微测或总测）
def get_total_score(df, col_start, col_end):
    df['总分'] = df.iloc[:, col_start:col_end].sum(axis=1)
    return df['总分']

# 将题目编码转换成核心概念
def replace_q_code_with_concept(general, coding):
    copy_df = general.copy(deep=True)
    copy_df.columns = list(map(lambda x: coding.loc[x]['核心概念'], general.columns))
    return copy_df

# 获得某一核心概念的总分
def get_concept_total_score(df_concept, concept):
    grouped = df_concept.groupby(df_concept.columns, axis=1)
    df = grouped.sum()
    return df[concept]

def tests_difficulty(test_list, concept):
    if len(test_list) == 0:
        raise ValueError('No tests at all.')

    tests = [test for test in test_list if test is not None]

    tests_name = []
    tests_difficulty = []
    for test in test_list:
        tests_name.append(test['name'])
        t_difficulty = test_difficulty(test['content'], test['type'], concept)
        tests_difficulty.append(t_difficulty)

    df = pd.DataFrame(tests_difficulty, index=tests_name, columns=['难度'])
    return df

# 一个测试的难度
def test_difficulty(test, test_type, concept):
    if test_type == 'micro':
        test = test.iloc[:, :-1] # 去掉作答日期一列
    elif test_type == 'general':
        test = test[concept] # 取出难度分析的核心概念的所有题目
    # 每道题的难度为该题所有学生的平均得分率的均值
    q_difficulties = test.mean(axis=0)
    if isinstance(q_difficulties, pd.Series):
        # 该测试的难度所有题的难度的均值
        t_difficulty = q_difficulties.mean()
    else:
        t_difficulty = q_difficulties # 总测可能只有一题是该核心概念的题

    return t_difficulty

# 测试的正太分布情况
def normal_distribution(df, test_name, bins):
    data = get_total_score(df)
    # Fit a normal distribution to the data:
    mu, std = norm.fit(data)

    # Plot the histogram.
    plt.hist(data, bins=bins, normed=True, alpha=0.6, color='g')

    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    title = test_name + ': $\mu = %.1f$,  $\sigma = %.2f$' % (mu, std)
    plt.title(title, fontproperties=chinesefont)
    plt.show()

if __name__ == "__main__": main()
