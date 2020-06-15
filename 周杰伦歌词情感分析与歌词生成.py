# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 11:00:19 2020

@author: lx
"""
import tensorflow as tf
from tensorflow import keras 
import pandas as pd
import numpy as np
import requests 
import json
import os 
import collections
import matplotlib.pyplot as plt
import jieba
import re
from PIL import Image
import wordcloud
from wordcloud import ImageColorGenerator
from sklearn.feature_extraction.text import CountVectorizer

###########################   数据获取----爬虫部分      #########################
os.getcwd()                              #查看当前路径
os.chdir('C:\\Users\\lx\\Desktop')       #更改路径至桌面
# 引用requests,json模块 
url = 'https://c.y.qq.com/soso/fcgi-bin/client_search_cp'
headers = { 
    'referer':'https://y.qq.com/portal/search.html',      #从qq音乐爬取数据
    # 请求来源 
    'user-agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6)AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36'
    # 标记了请求从什么设备，什么浏览器上发出 11

} 
for x in range(1): 
    params = {
        'ct':'24',
        'qqmusic_ver': '1298', 
        'new_json':'1', 
        'remoteplace':'sizer.yqq.lyric_next', 
        'searchid':'94267071827046963', 
        'aggr':'1',
        'cr':'1', 
        'catZhida':'1', 
        'lossless':'0', 
        'sem':'1', 
        't':'7', 
        'p':str(x+1), 
        'n':'60', 
        'w':'周杰伦', 
        'g_tk':'1714057807', 
        'loginUin':'0', 
        'hostUin':'0', 
        'format':'json', 
        'inCharset':'utf8', 
        'outCharset':'utf-8', 
        'notice':'0', 
        'platform':'yqq.json', 
        'needNewCode':'0' 
    }
    res = requests.get(url, params = params) 
    #下载该网页，赋值给res 
    jsonres = json.loads(res.text) 
    #使用json来解析res.text 
    list_lyric = jsonres['data']['lyric']['list'] 
    #一层一层地取字典，获取歌词的列表 
    for lyric in list_lyric: 
        # lyric是一个列表，x是它里面的元素 
        print(lyric['content']+'\n') 
        #以content为键，查找歌词 

#将结果保存在一个列表中#
result=[]
for lyric in list_lyric: 
        # lyric是一个列表，x是它里面的元素 
        result.append(lyric['content']) 
        #以content为键，查找歌词 
print(result)

#导出数据,保存为csv格式（不做任何处理）
Music=pd.core.frame.DataFrame(result)
Music.to_csv('Music.csv',encoding='gbk')    ##输出数据



###########################  第一部分 描述性统计分析  ###########################
## 数据可视化 ##
#数据导入
data = pd.read_csv("Music.csv")
data.head(5)

#分组聚合
Group1=data[['歌名','所属专辑']].groupby(by='所属专辑') #按专辑分
type(Group1)
Group1.sum() #各专辑下的名曲
#曲目统计
d1 = collections.Counter(data["所属专辑"]) # 瞬间出结果 


Group2=data[['歌名','年份']].groupby(by='年份') #按年份分
type(Group2)
Group2.sum() #各专辑下的名曲
#曲目统计
d2 = collections.Counter(data["年份"]) # 瞬间出结果

#然后可以做简单的可视化(尚未完成！！！！)
#专辑直方图
dic = d1
s = sorted(dic.items(), key=lambda x: x[1], reverse=False)  # 对dict 按照value排序 True表示翻转 ,转为了列表形式
print(s)
x_x = []
y_y = []
for i in s:
    x_x.append(i[0])
    y_y.append(i[1])

x = x_x
y = y_y

fig, ax = plt.subplots()
ax.barh(x, y, color="deepskyblue")
labels = ax.get_xticklabels()
plt.setp(labels, rotation=0, horizontalalignment='right')

for a, b in zip(x, y):
    plt.text(b+1, a, b, ha='center', va='center')
ax.legend(["label"],loc="lower right")

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.ylabel('The name of album')
plt.xlabel('Frequency')
plt.rcParams['savefig.dpi'] = 300  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率
plt.title("Heat of album")
plt.savefig('result1.png')
plt.show()

#年份直方图
dic = d2
s = sorted(dic.items(), key=lambda x: x[1], reverse=False)  # 对dict 按照value排序 True表示翻转 ,转为了列表形式
print(s)
x_x = []
y_y = []
for i in s:
    x_x.append(i[0])
    y_y.append(i[1])

x = x_x
y = y_y

fig, ax = plt.subplots()
ax.barh(x, y, color="deepskyblue")
labels = ax.get_xticklabels()
plt.setp(labels, rotation=0, horizontalalignment='right')

for a, b in zip(x, y):
    plt.text(b+1, a, b, ha='center', va='center')
ax.legend(["label"],loc="lower right")

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.ylabel('year')
plt.xlabel('Frequency')
plt.rcParams['savefig.dpi'] = 300  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率
plt.title("Heat of year")
plt.savefig('result2.png')
plt.show()

## 词频统计分析 ##
#导入停用词表
with open('stopwords.txt',encoding='gb18030', errors='ignore') as s:
    stopwords = set([line.replace('\n', '') for line in s])

#传入apply的预处理函数，完成中文提取、分词以及多余空格剔除
def preprocessing(c):

    c = [word for word in jieba.cut(''.join(re.findall('[\u4e00-\u9fa5]+', c))) if word != ''and word not in stopwords]

    return ' '.join(c)

#将所有语料按空格拼接为一整段文字
result = ' '.join(data['歌词'].apply(preprocessing))
result[1]

#词云图轮廓
ZJL_mask = np.array(Image.open('ZJL.png'))
image_colors = ImageColorGenerator(ZJL_mask)

#从文本中生成词云图
w=wordcloud.WordCloud(font_path='Simkai.ttf', # 定义SimHei字体文件
                      background_color='white', # 背景色为白色
                      mask=ZJL_mask, # 添加模板
                      height=400, # 高度设置为400
                      width=800, # 宽度设置为800
                      scale=20, # 长宽拉伸程度程度为20
                      relative_scaling=0.3, # 设置字体大小与词频的关联程度为0.3
                     )
w.generate(result)                     #生成
w.to_file('词云.png')                  #保存到本地


##########################   第二部分 歌词情感分析    ###########################
#逐首歌曲断词
def chinese_work_cut(mytest):
    return " ".join(jieba.cut(mytest))
data['已裁剪的歌词'] = data["歌词"].apply(chinese_work_cut)  #将结果拼到数据框后头

#snow库方法

#深度学习方法
#分为训练集与测试集
X = data["已裁剪的歌词"] 
y = data["情感(0消极、1中性、2积极)"]
X_train = X[10:]
y_train = y[10:]
X_test = X[:10]
y_test = y[:10]

#将歌词one-hot编码为向量
vect = CountVectorizer(max_df = 0.8,min_df = 3,token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b')  
ALL=pd.concat([X_train,X_test])
ALL = pd.DataFrame(vect.fit_transform(ALL).toarray(),columns=vect.get_feature_names())
ALL.head()
X_train = ALL[10:]
X_test = ALL[:10]
len(X_train.iloc[1])                     #查看词维度


#建模---构建三个隐层的DNN
model1 = keras.Sequential([
    keras.layers.Dense(30, activation='relu',input_shape=(len(X_train.iloc[1]) ,)),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])
model1.summary()

#设置loss与优化算法
model1.compile(optimizer=keras.optimizers.Adam(),
             loss="sparse_categorical_crossentropy",
             metrics=['accuracy'])

#开始训练!!
history = model1.fit(X_train, y_train, epochs=50, validation_split=0.1)  #0.1指交叉验证集占训练集的1/10！

#可视化学习曲线
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1.5)
    plt.show()
plot_learning_curves(history)

#验证集上表现差！！模型过拟合!!!重新训练,减少迭代次数!!!
history_2 = model1.fit(X_train, y_train, epochs=25, validation_split=0.1) 

#再次可视化学习曲线
plot_learning_curves(history_2)

#在训练集上效果还是不好,考虑简化模型!!! 去掉一层隐层同时减少节点!!!
model2 = keras.Sequential([
    keras.layers.Dense(25, activation='relu',input_shape=(len(X_train.iloc[1]) ,)),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])
model2.summary()

#设置loss与优化算法
model2.compile(optimizer=keras.optimizers.Adam(),
             loss="sparse_categorical_crossentropy",
             metrics=['accuracy'])

#开始训练!!
history3 = model2.fit(X_train, y_train, epochs=20, validation_split=0.1) 

#再次可视化学习曲线
plot_learning_curves(history3)

#查看在测试集上的表现
model2.evaluate(X_test,y_test) #测试误差达到百分之70%!!

#查看具体预测结果
predict = model2.predict(np.array(X_test))
predict_value = pd.DataFrame({'预测情感':[2,1,0,0,1,0,2,1,0,1]})
predict_result = pd.concat([pd.DataFrame(predict),data[:10][["歌名","歌词","情感(0消极、1中性、2积极)"]],predict_value],axis=1)


#建模---构建RNN
embedding_dim = 16
model2 = keras.Sequential([
    keras.layers.Embedding(10, embedding_dim, input_length = len(X_train.iloc[1])),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])
model2.summary()

##########################    第三部分 歌词生成器    ############################

