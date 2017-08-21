import re
import  pandas as pd

# 试卷编码【新编码】
def clean_new_form_coding(coding):
    coding_columns = ['题目编码', '得分点编码', '一级主题代码', '一级主题', '二级主题代码', '二级主题', '核心概念代码', '核心概念', '学习表现指标代码', '学习表现指标',
                      '核心素养代码', '核心素养', '内容属性代码', '内容属性', '熟悉度代码', '熟悉度', '间接度代码', '间接度', '评分标准']
    if len(coding.columns) == 20:
        coding_columns.append( '能力水平')
    coding.columns = coding_columns
    coding = coding[['题目编码', '学习表现指标代码', '核心概念', '核心素养', '评分标准']]
    coding['题目编码'] = coding['题目编码'].map(lambda x: '0' + str(x))
    coding['评分标准'] = coding['评分标准'].map(lambda x: get_score(x))
    coding.rename(columns={'评分标准': '总分'}, inplace=True)
    coding.set_index(['题目编码'], inplace=True)
    return coding

def clean_old_form_coding(coding):
    coding.columns = ['题目编码', '学习表现指标代码', '核心素养', '评分标准']
    coding = coding.iloc[1:]
    coding['题目编码'] = coding['题目编码'].apply(lambda x: 'P0' + str(x) + '01')
    coding['评分标准'] = coding['评分标准'].apply(lambda x: get_score(x))
    coding.rename(columns={'评分标准': '总分'}, inplace=True)
    coding.set_index(['题目编码'], inplace=True)
    return coding

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

def add_date(df, filename_date):
    df_with_date = pd.read_excel(filename_date, converters={'学号':str, '作答日期':pd.to_datetime})
    df_with_date = df_with_date[['学号', '姓名', '作答日期']]
    df = pd.merge(df, df_with_date, left_on=['学号', '姓名'], right_on=['学号', '姓名'], how='left')
    return df