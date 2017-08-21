import pandas as pd
import SmartPartner.code.general_micro.utility as utility

def main():
    #general01, coding01, general05, coding05, general07, coding07 = general_test('D:/PycharmProjects/SmartPartner/data/general_test/math/')
    micro1, coding1, micro2, coding2, micro3, coding3, micro4, coding4, micro5, coding5 = micro_test('D:/PycharmProjects/SmartPartner/data/micro_test/math/')
    print(coding3)

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
    df1 = clean_micro_test(df1, file_path + '/with-date/2016-数学-八年级-上学期-单元微测-001（分式1）-date.xlsx')
    df2 = clean_micro_test(df2, file_path + '/with-date/2016-数学-八年级-上学期-单元微测-002（分式2）-date.xlsx')
    df3 = clean_micro_test(df3, file_path + '/with-date/2016-数学-八年级-上学期-单元微测-001（二次根式1）-date.xlsx')
    df4 = clean_micro_test(df4, file_path + '/with-date/2016-数学-八年级-上学期-单元微测-002（二次根式2）-date.xlsx')
    df5 = clean_micro_test(df5, file_path + '/with-date/2016-数学-八年级-下学期-单元微测-001（变量之间的关系）-date.xlsx')

    coding1 = utility.clean_new_form_coding(coding1)
    coding2 = utility.clean_new_form_coding(coding2)
    coding3 = utility.clean_old_form_coding(coding3)
    coding4 = utility.clean_old_form_coding(coding4)
    coding5 = utility.clean_new_form_coding(coding5)

    return df1, coding1, df2, coding2, df3, coding3, df4, coding4, df5, coding5

def read_data_from_micro_test(filename):
    df = pd.read_excel(filename, sheetname=1, converters={'学号': str})
    coding = pd.read_excel(filename, sheetname=2, skiprows=1)
    return df, coding

def clean_micro_test(df, filename_date):
    # add test date
    df = utility.add_date(df, filename_date)
    # tidy data
    df = df.iloc[:, 6:]
    df.drop('性别', axis=1, inplace=True)
    df.rename(columns={'学号': '教育ID'}, inplace=True)
    df.set_index(['教育ID', '姓名'], inplace=True)
    return df

if __name__ == "__main__": main()
