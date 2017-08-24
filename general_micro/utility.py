import re
import  pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
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
        coding['题目编码'] = coding['题目编码'].map(lambda x: '0' + str(x))
        coding['评分标准'] = coding['评分标准'].map(lambda x: get_score(x))
        coding.rename(columns={'评分标准': '总分'}, inplace=True)
        coding.set_index(['题目编码'], inplace=True)
        codings[i] = coding

    return codings

# 试卷编码【旧编码】
def clean_old_form_coding(coding):
    coding.columns = ['题目编码', '学习表现指标代码', '核心素养', '评分标准']
    coding['题目编码'] = coding['题目编码'].apply(lambda x: '0' + str(x))
    coding['评分标准'] = coding['评分标准'].apply(lambda x: get_score(x))
    coding.rename(columns={'评分标准': '总分'}, inplace=True)
    coding.set_index(['题目编码'], inplace=True)
    return coding

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
    return score

# 增加微测作答日期
def add_date(df, filename_date):
    df_with_date = pd.read_excel(filename_date, converters={'学号':str, '作答日期':pd.to_datetime})
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
    # add test date
    df = add_date(df, filename_date)
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

def plot_bar_chart(df, title):
    ax = df.plot(kind='bar', legend=True)
    ax.set_ylim(0,1)
    ax.set_title(title, fontproperties=chinesefont)
    ax.set_xticklabels(df.index.values, fontproperties=chinesefont)
    plt.legend(prop=chinesefont)
    plt.show()