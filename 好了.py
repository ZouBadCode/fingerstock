import cv2
import mediapipe as mp
import math
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import pyautogui

# Download the stock data
print('請輸入國家,美國,.TW:台灣上市,.TWO:台灣上櫃')
country_code = input('輸入國家地區')
ticker = input('輸入股票代碼')
stock_data = yf.download(ticker + country_code, period='1y', interval='1d')


# Clean and format the data
data = stock_data.reset_index()
data.columns = ['現在時間', '開盤價', '最高價', '最低價', '收價', '調整後收盤價', '成交量']
data['現在時間'] = pd.to_datetime(data['現在時間'].dt.strftime('%Y-%m-%d %H:%M'))
data['成交量'] //= 1000  # 把單位從成交股數換成成交張數


# Calculate the RSI
n = 14  # Number of periods to consider for the RSI calculation
data['Change'] = data['收價'].pct_change(
) * 100  # Calculate the change in price as a percentage
data['Gain'] = data['Change'].apply(lambda xl: xl
                                    if xl > 0 else 0)  # Calculate the gains
data['Loss'] = data['Change'].apply(lambda xl: abs(xl)
                                    if xl < 0 else 0)  # Calculate the losses


# Calculate the average gain and loss over the n periods
data['Avg Gain'] = data['Gain'].rolling(n).mean()
data['Avg Loss'] = data['Loss'].rolling(n).mean()


# Calculate the RSI as the ratio of the average gain to the average loss
data['RSI'] = data['Avg Gain'] / (data['Avg Loss'] + data['Avg Gain']) * 100


# Create the figure object
fig = go.Figure()


# Add the RSI trace
fig.add_trace(
    go.Scatter(x=data['現在時間'], y=data['RSI'], name='RSI', mode='lines'))


# Update the layout
fig.update_layout(title=ticker + country_code,
                  hovermode='x unified',
                  yaxis=dict(title='RSI'),
                  font=dict(size=20))
fig.update_yaxes(fixedrange=True)


config = dict({'scrollZoom': True})
fig.show(config=config)


angle = 0





def scroll_wheel_based_on_x(new_angle,old_angle):

    if new_angle > old_angle:
        pyautogui.scroll(1)  # 向上滚动
    elif new_angle < old_angle:
        pyautogui.scroll(-1)  # 向下滚动
    else:
        return  # 如果X的值未变化，则不进行滚动
while True:
    old_angle = 0
    new_angle = angle # 获取新的X值
    old_angle = new_angle
    scroll_wheel_based_on_x(new_angle,old_angle)









def calculate_angle(v1, v2):
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    magnitude_v1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    magnitude_v2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    cosine_angle = dot_product / (magnitude_v1 * magnitude_v2)
    angle = math.degrees(math.acos(cosine_angle))
    return angle

def detect_hand_angle():
    global angle
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75
    )
    
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                keypoint_pos = []
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    keypoint_pos.append((x, y))
                
                if len(keypoint_pos) >= 13:
                    # 計算食指和中指之間的向量
                    v_index = (keypoint_pos[8][0] - keypoint_pos[6][0], keypoint_pos[8][1] - keypoint_pos[6][1])
                    v_middle = (keypoint_pos[12][0] - keypoint_pos[10][0], keypoint_pos[12][1] - keypoint_pos[10][1])
                    
                    # 計算食指和中指之間的角度
                    angle = calculate_angle(v_index, v_middle)
                    
                    # 顯示食指和中指之間的角度
                    cv2.putText(frame, f"Index-Middle Angle: {angle:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Hand Angle Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_hand_angle()
