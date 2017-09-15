import re
import  pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import pearsonr
from scipy.stats import spearmanr
chinesefont = font_manager.FontProperties()
chinesefont.set_family('SimHei')

# 试卷编码【新编码】
def clean_new_form_coding(coding_list):
    if len(coding_list) == 0:
        raise ValueError('No codings at all.')

    codings = [coding for coding in coding_list if coding is not None]

    coding_columns = ['题目编码', '得分点编码', '一级主题代码', '一级主题', '二级主题代码', '二级主题', '核心概念代码', '核心概念', '学习表现指标代码', '学习表现指标',
                      '核心素养代码', '核心素养', '内容属性代码', '内容属性', '熟悉度代码', '熟悉度', '间接度代码', '间接度', '评分标准']
    coding_columns_with_ablity_level = list(coding_columns)
    coding_columns_with_ablity_level.append('能力水平')

    for i, coding in enumerate(codings):
        if len(coding.columns) == 20:
            coding.columns = coding_columns_with_ablity_level
        else:
            coding.columns = coding_columns
        coding = coding[['题目编码', '学习表现指标代码', '核心概念', '核心素养', '评分标准']]
        new_coding = coding.copy() # 防止出现set
        new_coding.loc[:,'题目编码'] = new_coding.loc[:, '题目编码'].apply(lambda x: '0' + str(x))
        new_coding.loc[:,'总分'] = new_coding.loc[:, '评分标准'].apply(lambda x: get_score(x))
        #coding.rename(columns={'评分标准': '总分'}, inplace=True)
        new_coding.set_index(['题目编码'], inplace=True)
        codings[i] = new_coding

    # 只有一个元素，不返回数组，而是返回一个对象
    if len(codings) == 1:
        codings = codings[0]
    return codings

# 试卷编码【旧编码】
def clean_old_form_coding(coding):
    coding.columns = ['题目编码', '学习表现指标代码', '核心素养', '评分标准']
    new_coding = coding.copy()
    new_coding.loc[:,'题目编码'] = new_coding.loc[:, '题目编码'].apply(lambda x: '0' + str(x))
    new_coding.loc[:,'总分'] = new_coding.loc[:, '评分标准'].apply(lambda x: get_score(x))
    #coding.rename(columns={'评分标准': '总分'}, inplace=True)
    new_coding.set_index(['题目编码'], inplace=True)
    return new_coding

# 评分标准获取题目总分
def get_score(x):
    score = x[x.index('【'):x.index('】')+1]
    comma = re.search(',|，', score)
    # 百分比总分
    if comma != None:
        score = score.split(comma.group(0))[1]
        score = score[0:score.index('分')]
    else:
        score = score[1:score.index('分')]

    return int(score)

# 增加微测作答日期
def add_date(df, filename_date):
    df_with_date = pd.read_excel(filename_date, converters={'学号':str, '作答日期':pd.to_datetime})
    # print('date:')
    # print(df_with_date.shape)
    df_with_date = remove_duplicate_record(df_with_date)
    # print('date:')
    # print(df_with_date.shape)
    df_with_date = df_with_date[['学号', '姓名', '作答日期']]
    df = pd.merge(df, df_with_date, left_on=['学号', '姓名'], right_on=['学号', '姓名'], how='left')
    return df

# 平均得分率
def avg_score(score_df, code_df):
    copy_df = score_df.copy(deep=True)
    for colname, col in copy_df.iteritems():
        if colname != '作答日期':
            score = code_df.loc[colname]['总分']
            copy_df[colname] = copy_df[colname].map(lambda x: x / int(score))
    return copy_df

# 读微测数据
def read_data_from_micro_test(filename):
    df = pd.read_excel(filename, sheetname=1, converters={'学号': str})
    coding = pd.read_excel(filename, sheetname=2, skiprows=1)
    return df, coding

# 清洗调整微测数据
def clean_micro_test(df, filename_date, q_code_subtract_way):
    df = add_date(df, filename_date)
    df = df.iloc[:, 6:]
    df.drop('性别', axis=1, inplace=True)
    df.rename(columns={'学号': '教育ID'}, inplace=True)
    df.set_index(['教育ID', '姓名'], inplace=True)
    if q_code_subtract_way == 'normal':
        df.columns.values[:-1] = list(map(lambda x: x[1:-2], df.columns[:-1]))
    else: # 有的微测如micro4编码表对应的编码不是P之后的六位数值，而是P之后的四位数值加上后两位数值
        df.columns.values[:-1] = list(map(lambda x: x[1:5] + x[-2:], df.columns[:-1]))
    return df

# 去掉一个学生做多次微测的记录
def remove_duplicate_record(df):
    #df = df.groupby(df.index).first()
    df = df.drop_duplicates(subset='学号', keep='first')
    return df

def plot_bar_chart(df, title):
    ax = df.plot(kind='bar', legend=True)
    ax.set_ylim(0,1)
    ax.set_title(title, fontproperties=chinesefont)
    ax.set_xticklabels(df.index.values, fontproperties=chinesefont)
    plt.legend(prop=chinesefont)
    plt.show()

# 测试成绩的相关性分析
def corr_analysis(df_x, df_y, concept, objs):
    if objs == 'general-micro':
        series_x = get_concept_total_score(df_x, concept) # 总测该概念的总得分
        series_y = get_total_score(df_y, 0, len(df_y.columns)-1) # 微测的所得总分
    elif objs == 'micro-micro':
        series_x = get_total_score(df_x, 0, len(df_x.columns) - 1)
        series_y = get_total_score(df_y, 0, len(df_y.columns) - 1)
    df = pd.merge(series_x.to_frame(), series_y.to_frame(), left_index=True, right_index=True) # 两测试都参加的学生
    x = df.iloc[:, 0]
    y = df.iloc[:, 1]
    #corr, p_value = pearsonr(x, y)
    corr, p_value = spearmanr(x, y)
    print(corr, p_value)
    return corr, p_value

# 计算总分（微测或总测）
def get_total_score(df, col_start, col_end):
    df['总分'] = df.iloc[:, col_start:col_end].sum(axis=1)
    return df['总分']

# 获得试卷中某一核心概念的总分
def get_concept_total_score(df_concept, concept):
    grouped = df_concept.groupby(df_concept.columns, axis=1)
    df = grouped.sum()
    return df[concept]

# 获得编码表里某一核心概念的总分
def get_total_score_from_coding(coding, concept=''):
    if concept == '':
        total_score = coding['总分'].sum()
    else:
        grouped = coding.groupby('核心概念')
        total_score = grouped['总分'].sum()
    return total_score

# 将题目编码转换成核心概念
def replace_q_code_with_concept(general, coding):
    copy_df = general.copy(deep=True)
    copy_df.columns = list(map(lambda x: coding.loc[x]['核心概念'], general.columns))
    return copy_df

def tests_difficulty(test_list, concept):
    if len(test_list) == 0:
        raise ValueError('No tests at all.')

    tests = [test for test in test_list if test is not None]

    tests_name = []
    tests_difficulty = []
    qestions_difficulty = []
    for test in test_list:
        tests_name.append(test['name'])
        t_difficulty, q_difficulties = test_difficulty(test['content'], test['type'], concept)
        tests_difficulty.append(t_difficulty)
        qestions_difficulty.append({'name':test['name'], 'q_difficulty':q_difficulties})

    df = pd.DataFrame(tests_difficulty, index=tests_name, columns=['难度'])
    return df, qestions_difficulty

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

    return t_difficulty, q_difficulties

def output_questions_difficulty(obj_list, filename):
    index = []
    data = []
    for obj in obj_list:
        index.append(obj['name'])
        difficulties = obj['q_difficulty']
        if isinstance(difficulties, pd.Series):
            difficulties = difficulties.tolist()
        else: # 该核心概念只有一题
            difficulties = [difficulties]
        data.append(difficulties)
    df = pd.DataFrame(data, index=index)
    df.to_csv('D:/PycharmProjects/SmartPartner/data/output/q_difficulty_'+filename+'.csv')

def to_rank_score(df, coding):
    rank_origin_dict = get_rank_origin_score(coding)

def get_rank_origin_score(coding):
    coding['评分标准'] = coding['评分标准'].apply(lambda x: re.findall('【(.*?)】', x))
    print(coding)

# 考总测前参加微测的学生在总测的表现
def general_micro_score_compare_on_time(concept_avg_general, general_name, micro_list, concept, start_time, end_time):
    general_total_score = get_concept_avg_score(concept_avg_general, concept)
    for i, micro in enumerate(micro_list):
        avg_micro_before, avg_micro_after = get_micro_time_groups(micro, start_time, end_time, concept, i+1)
        if i == 0:
            df = pd.merge(general_total_score.to_frame(), avg_micro_before, left_index=True, right_index=True,
                          how='left')
        else:
            df = pd.merge(df, avg_micro_before, left_index=True, right_index=True, how='left')

    df.to_csv('D:/PycharmProjects/SmartPartner/data/output/' + general_name + '总测前参加微测的学生在总测的表现--' + concept + '.csv')

# 将微测分为考试前和考试后两组
def get_micro_time_groups(avg_micro, start_time, end_time, concept, index):
    avg_micro['总分'] = avg_micro.iloc[:, 0:len(avg_micro.columns) - 1].mean(axis=1)
    avg_micro_score_date = avg_micro[['总分', '作答日期']]
    avg_micro_score_date.rename(columns={'总分': concept + '微测'+ str(index)}, inplace=True)
    avg_micro_before = avg_micro_score_date[avg_micro_score_date['作答日期'] < start_time]
    avg_micro_after = avg_micro_score_date[avg_micro_score_date['作答日期'] > end_time]
    return avg_micro_before, avg_micro_after

def get_concept_avg_score(df_concept, concept):
    grouped = df_concept.groupby(df_concept.columns, axis=1)
    df = grouped.mean()
    return df[concept]