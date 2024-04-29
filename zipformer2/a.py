# 定义文件名
filename = 'convert.txt'

# 初始化一个空字符串用于存储结果
result = ''

# 打开文件并读取每一行
with open(filename, 'r', encoding='utf-8') as file:
    for line in file:
        # 分割每一行，取最后一个单词
        last_word = line.strip().split(' ')[-1]
        # 将单词添加到结果字符串中，并加上逗号
        result += last_word + ','

# 去除最后一个多余的逗号
result = result.rstrip(',')

# 打印结果
print("["+result+"]")