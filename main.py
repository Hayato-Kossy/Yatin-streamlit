import pandas as pd
import numpy as np
import math

from sklearn.preprocessing import LabelEncoder

import streamlit as st
import lightgbm as lgb
from lightgbm.engine import train

#StreamLitのUI関連
st.title('あなたの家賃High&Low〜in23区〜')


st.sidebar.write(f"""
## 質問に答えてください *単位や書式を守ってください！！
""")

#StreamLitからquestion_dfを取得
#質問項目から導く変数

Place = st.sidebar.text_input('任意の23区の住所（出来れば番地まで、なくてもなんとかなる）', '例:東京都目黒区〇〇x-y-z')
Passed = st.sidebar.text_input('築年数','例:50年')
Area = st.sidebar.text_input('面積','例:50.8m2')
Floor = st.sidebar.text_input('所在階','例:1階／12階建')
Parking = st.sidebar.selectbox('駐車場',('駐車場有', '駐車場なし'))
st.sidebar.write('ここから下は任意の条件を残してください')
Bath = st.sidebar.text_input('風呂、トイレ','専用バス／\t専用トイレ／\tバス・トイレ別／\tシャワー／\t追焚機能／\t温水洗浄便座')
Kitchen = st.sidebar.text_input('キッチン','ガスコンロ／\tコンロ3口／\tシステムキッチン\t／\t給湯／\tL字キッチン')
Facility = st.sidebar.text_input('設備','エアコン付\tシューズボックス／\tバルコニー／\tフローリング／\tエレベータ')
Your_Yatin = st.sidebar.slider('家賃',0,400000,60000)
Start = st.sidebar.button('決定')


question_df = pd.DataFrame({
        'Place'     :[Place],	
        'Passed'	:[Passed],
        'Area'	    :[Area],
        'Floor'     :[Floor],	
        'Parking'   :[Parking],
        'Bath'      :[Bath],
        'Kitchen'   :[Kitchen],
        'Facility'  :[Facility]
    })

st.write(f"""# あなたのデータ↓""")
st.table(question_df)



#学習データの取得
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')



train = train.rename(columns={'賃料':'target', '契約期間':'Contract', '間取り':'Room', 
                              '築年数':'Passed', '駐車場':'Parking', '室内設備':'Facility', 
                              '放送・通信':'Internet', '周辺環境':'Building', '建物構造':'Material', 
                              '面積':'Area', 'キッチン':'Kitchen', '所在地':'Place',
                              'バス・トイレ':'Bath', '所在階':'Floor', 'アクセス':'Access', 
                              '方角':'Angle'})
test = test.rename(columns={'契約期間':'Contract', '間取り':'Room', '築年数':'Passed', 
                            '駐車場':'Parking', '室内設備':'Facility', '放送・通信':'Internet', 
                            '周辺環境':'Building', '建物構造':'Material', '面積':'Area', 
                            'キッチン':'Kitchen', '所在地':'Place', 'バス・トイレ':'Bath', 
                            '所在階':'Floor', 'アクセス':'Access', '方角':'Angle'})
#質問にすることが困難または処理が難しい変数は削除
train = train.drop(['id','Access', 'Room','Building','Angle','Internet','Material','Contract'], axis=1)
test = test.drop(['id','Access', 'Room','Building','Angle','Internet','Material','Contract'], axis=1)

test = pd.concat([question_df, test])
X_test = pd.concat([question_df, test])

#特徴量の作成
def makeCountFull(train, test, categorical_features=None, report=True):
    add_cols = categorical_features
    if report:
        print('add_cols: ', add_cols)
    for add_col in add_cols:
        train[add_col + '_countall'] = train[add_col].map(pd.concat([train[add_col], test[add_col]], ignore_index=True).value_counts(dropna=False))
        test[add_col + '_countall'] = test[add_col].map(pd.concat([train[add_col], test[add_col]], ignore_index=True).value_counts(dropna=False))
    return train, test
cat_features = ['Place', 'Passed', 'Area', 'Floor', 'Bath', 'Kitchen', 'Facility', 'Parking']
train, test = makeCountFull(train, test, cat_features)   

#カテゴリカル変数のラベルエンコーディング
cat_cols = ['Place', 'Passed', 'Area', 'Floor', 'Parking','Bath',
            'Kitchen', 'Facility']
for col in cat_cols:
    train[col] = train[col].astype(str)
    test[col] = test[col].astype(str)
    le = LabelEncoder()
    le.fit(list(train[col])+list(test[col]))
    train[col] = le.transform(train[col])
    test[col]    = le.transform(test[col])    
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')

#実行前段階の処理
X = train.drop(['target'], axis=1)
y = train['target']


#'''X'''
#st.table(X.head())
#'''y'''
#st.table(y.head())

#実行フェーズ
if Start is True:

    #LightGBMでモデル作成
    cv_score = 0
    param = {'n_estimators': list(range(10, 30, 10)),
             'learning_rate': list(np.arange(0.05, 0.20, 0.01))}

    model = lgb.LGBMRegressor(num_leaves=100,learning_rate=0.05,n_estimators=1000)
    model.fit(X,y)

    y_pred = model.predict(test)
    AI_Yatin = math.floor(y_pred[0])

    st.write(f"""## AIが出した家賃は""" +str(AI_Yatin) + f""" 円です""")
    
    if Your_Yatin > AI_Yatin:
        st.write('あなたは損してますね')
    elif Your_Yatin < AI_Yatin:
        st.write('あなたは得をしているかもしれませんね')
    else:
        st.write('予想とぴったり')


