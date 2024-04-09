import cv2
import mediapipe as mp
import math
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import pyautogui
import numpy as np


print('請輸入國家,美國,.TW:台灣上市,.TWO:台灣上櫃')
country_code = input('輸入國家地區')
ticker = input('輸入股票代碼')
stock_data = yf.download(ticker + country_code, period='1y', interval='1d')


data = stock_data.reset_index()
data.columns = ['現在時間', '開盤價', '最高價', '最低價', '收價', '調整後收盤價', '成交量']
data['現在時間'] = pd.to_datetime(data['現在時間'].dt.strftime('%Y-%m-%d %H:%M'))
data['成交量'] //= 1000  # 把單位從成交股數換成成交張數



n = 14  
data['Change'] = data['收價'].pct_change(
) * 100  
data['Gain'] = data['Change'].apply(lambda xl: xl
                                    if xl > 0 else 0)  
data['Loss'] = data['Change'].apply(lambda xl: abs(xl)
                                    if xl < 0 else 0)  



data['Avg Gain'] = data['Gain'].rolling(n).mean()
data['Avg Loss'] = data['Loss'].rolling(n).mean()



data['RSI'] = data['Avg Gain'] / (data['Avg Loss'] + data['Avg Gain']) * 100



fig = go.Figure()



fig.add_trace(
    go.Scatter(x=data['現在時間'], y=data['RSI'], name='RSI', mode='lines'))



fig.update_layout(title=ticker + country_code,
                  hovermode='x unified',
                  yaxis=dict(title='RSI'),
                  font=dict(size=20))
fig.update_yaxes(fixedrange=True)


config = dict({'scrollZoom': True})
fig.show(config=config)








# 初始化Mediapipe手部解決方案
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

def calculate_angle(p1, p2, p3):
    # 計算三點形成的角度
    ba = p1 - p2
    bc = p3 - p2
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

# 初始化視訊攝影機
cap = cv2.VideoCapture(0)

old_angle = None

while True:
    success, image = cap.read()
    if not success:
        break

    # 轉換BGR圖像到RGB          
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 進行手部偵測
    results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 獲取食指與中指的關節點座標
            index_finger_tip = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                                         hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y])
            middle_finger_tip = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x,
                                          hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y])
            middle_finger_base = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x,
                                           hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y])

            # 計算角度
            angle = calculate_angle(index_finger_tip, middle_finger_base, middle_finger_tip)

            # 比較新舊角度並輸出結果
            if old_angle is not None:
                if angle < 45:
                    pyautogui.scroll(-1)  # 向下滚动
                elif angle > 45:
                    pyautogui.scroll(1)  # 向上滚动
            old_angle = angle

    # 顯示結果
    cv2.imshow('Frame', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
