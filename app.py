from flask import Flask, render_template, url_for, redirect, flash, Response, request
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError, DataRequired, EqualTo
from flask_bcrypt import Bcrypt
import webbrowser
import numpy as np
import os, cv2
import math
import mediapipe as mp

from PIL import Image
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from res.function.label_fuc import getCenter, getValue, getPoint
from res.function.converse_label import string_to_point
from res.function.show import show_point 
from res.function.label import detect_label

app = Flask(__name__)
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'thisisasecretkey'


login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'




@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)
    user_point = db.Column(db.String(50), nullable=False)
    def __init__(self, username, password, user_point):
        self.username =username
        self.password = password
        self.user_point = user_point
        
class RegisterForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "使用者名稱"})

    password = PasswordField(validators=[InputRequired(), Length(min=6, max=20),EqualTo('pass_confirm', message='密碼需要吻合')], render_kw={"placeholder": "密碼"})
    #password = PasswordField('密碼', validators=[DataRequired(), EqualTo('pass_confirm', message='密碼需要吻合')])
    
    pass_confirm = PasswordField(validators=[InputRequired(), Length(min=6, max=20)], render_kw={"placeholder": "確認密碼"})
    submit = SubmitField('註冊')
    def validate_username(self, username):
        existing_user_username = User.query.filter_by(
            username=username.data).first()
        if existing_user_username:
            raise ValidationError(
                '該名稱已有使用者註冊，請輸入新的名稱')
            


class LoginForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "使用者名稱"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=6, max=20)], render_kw={"placeholder": "密碼"})

    submit = SubmitField('登入')


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                flash("您已經成功的登入系統")
                return redirect(url_for('select'))
    return render_template('login.html', form=form)

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed_password, user_point = "NONE")
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

# @app.route('/add_items_db',methods=['GET','POST'])
# def add_items_db():
#     """加入 DB"""
#     form = AddItemForm()
#     if form.validate_on_submit():
#     #item
#     input_item = form.item.data
#     item_in_db = Configuration.query.filter_by(item = input_item).first()
#     if item_in_db:
#         flash('您填寫資料的資料已經存在！！!')
#     else:
#         configuration = Configuration(item=input_item)
#         db.session.add(configuration)
#         db.session.commit()
#         flash('感謝填寫資料！！！!')
#         return redirect(url_for('home'))
# return render_template('add_items_form.html',form=form)


###############################################################################################################################
@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    init_predict()
    logout_user()
    return redirect(url_for('login'))
@app.route('/select', methods=['GET', 'POST'])
@login_required
def select():
    init_predict()
    user = load_user(current_user.id)
    return render_template('Select.html', welcome_text = "歡迎您的歸來, " + user.username + "!!", label_text = "您目前紀錄的位置是: " + user.user_point)
###############################################################################################################################
numClasses = 31 #categories
model = None
camera_mode = False
global frame
frame = None
global mode
mode = 1
def load_data(cls_file):
    global classes, performance
    with open(cls_file, "r") as f:
        classes = f.read().split("\n")
    

def load_model(path):
    global model
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = (3,3), input_shape = (50, 50, 3), activation = 'relu'))
    model.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
    model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(512, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(numClasses, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
    try:
        model.load_weights('model/model.h5')
        print("load successed")
    except:
        print("load failed")
def reshape_input(photo):
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(min_detection_confidence=0.01, min_tracking_confidence=0.5, max_num_hands = 1)
    result = hands.process(photo)
    if result.multi_hand_landmarks:
        imgHeight = photo.shape[0]
        imgWidth = photo.shape[1]
        key_points = []
        most_top = 9999      
        
        for handLms in result.multi_hand_landmarks: 
            for lm in handLms.landmark:
                xPos = round(lm.x * imgWidth)
                yPos = round(lm.y * imgHeight)
                key_points.append([xPos, yPos])
                if yPos < most_top:
                    most_top = yPos
        dis = ((key_points[9][0] - key_points[0][0])**2 + (key_points[9][1] - key_points[0][1])**2)**0.5 
        new_width = int(2.5 * (dis))
        new_height = int((key_points[0][1] - most_top) + (2.5 * dis))
        start_x = key_points[0][0] - int(new_width / 2)
        if start_x < 0:
            start_x = 0
        end_x = start_x + new_width
        if end_x > imgWidth:
            end_x = imgWidth
        start_y = most_top - int(0.5 * dis)
        if start_y < 0:
            start_y = 0
        end_y = start_y + new_height
        
        if end_y > imgHeight:
            end_y = imgHeight
        return (photo[start_y : end_y, start_x : end_x], True)
    else:
        return (photo, False)
def reshape_photo(photo):
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(min_detection_confidence=0.01, min_tracking_confidence=0.5, max_num_hands = 1)
    result = hands.process(photo)
    if result.multi_hand_landmarks:
        imgHeight = photo.shape[0]
        imgWidth = photo.shape[1]
        key_points = []
        most_top = 9999      
        
        for handLms in result.multi_hand_landmarks: 
            for lm in handLms.landmark:
                xPos = round(lm.x * imgWidth)
                yPos = round(lm.y * imgHeight)
                key_points.append([xPos, yPos])
                if yPos < most_top:
                    most_top = yPos
        dis = ((key_points[9][0] - key_points[0][0])**2 + (key_points[9][1] - key_points[0][1])**2)**0.5 
        new_width = int(2.5 * (dis))
        new_height = int((key_points[0][1] - most_top) + (1.7 * dis))
        start_x = key_points[0][0] - int(new_width / 2)
        if start_x < 0:
            start_x = 0
        end_x = start_x + new_width
        if end_x > imgWidth:
            end_x = imgWidth
        start_y = most_top - int(0.2 * dis)
        if start_y < 0:
            start_y = 0
        end_y = start_y + new_height
        
        if end_y > imgHeight:
            end_y = imgHeight
        return (photo[start_y : end_y, start_x : end_x], True)
    else:
        return (photo, False)
def frames():
    while True:
        success, frame = camera.read()
        if success:
            try:
                
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
        else:
            pass
###############################################################################################################################
def init_predict():
    global camera_mode, mode, frame, show_pic, select
    camera_mode = False
    mode = 1
    frame = None
    show_pic = None
    select = 1
    release_camera()
def release_camera():
    try:
        camera.release()
        cv2.destroyAllWindows()
    except:
        pass
@app.route('/predict_home')
@login_required
def predict_home():
    init_predict()
    return render_template('page.html')

@app.route('/video_feed')
@login_required
def video_feed():
    global camera_mode
    if camera_mode == True:
        return Response(frames(), mimetype = 'multipart/x-mixed-replace; boundary=frame')
    else:
        try: 
            tmp = show_pic
            if tmp.shape[0] != 500: #reshape by rate
                tmp = cv2.resize(tmp, None, fx = 500 /tmp.shape[0],fy= 500 /tmp.shape[0],interpolation=cv2.INTER_LINEAR)
            ret, buffer = cv2.imencode('.jpg', cv2.flip(tmp, 1))
            show = buffer.tobytes()
            
        except:
            tmp = cv2.imread('static/preshow.png')
            ret, buffer = cv2.imencode('.jpg', tmp)
            show = buffer.tobytes()
            
        return Response(show, mimetype = 'multipart/x-mixed-replace;')

@app.route("/setCamera", methods=["POST"])
@login_required
def setCamera():
    global camera, camera_mode, mode
    if request.method == 'POST':
        mode = 0
        try:
            camera.release()
            cv2.destroyAllWindows()
        except:
            pass
        camera_mode = True
        camera = cv2.VideoCapture(0)
    if select == 1:
        return render_template('page.html')
    if select == 2: 
        user = load_user(current_user.id)
        return render_template('record_home.html', show_display_area = '您目前的儲存的位置為: ' + user.user_point)
    if select == 3:
        user = load_user(current_user.id)
        return render_template('predictByrecord_home.html', show_display_area = '您目前的儲存的位置為: ' + user.user_point)

@app.route("/choose_file", methods=["POST"])
@login_required
def choose_file():
    global camera_mode, show_pic, mode, file
    release_camera()
    if request.method == "POST":
        mode = 1
        camera_mode = False
        file = request.files['photo']
        if file.filename == '':
            flash('No image selected for uploading')
            return render_template('page.html')
        
        file = Image.open(file.stream).convert('RGB')
        file = cv2.flip(np.array(file), 1)
        print("y",file.shape[0])
        if file.shape[0] != 600: #reshape by rate
            file = cv2.resize(file, None,fx = 600 /file.shape[0],fy= 600 /file.shape[0],interpolation=cv2.INTER_LINEAR)
            print("resize", file.shape[1],file.shape[0])
        show_pic = cv2.cvtColor(file, cv2.COLOR_RGB2BGR)
        if select == 1:
            return render_template('page.html')
        if select == 2: 
            user = load_user(current_user.id)
            return render_template('record_home.html', show_display_area = '您目前的儲存的位置為: ' + user.user_point)
        if select == 3:
            user = load_user(current_user.id)
            return render_template('predictByrecord_home.html', show_display_area = '您目前的儲存的位置為: ' + user.user_point)
@app.route("/predict", methods=["POST"])
@login_required
def predict():
    global camera_mode, show_pic
    if request.method == "POST":
        if mode == 0: #use camera
            ret, img = camera.read()
            camera_mode = False
            camera.release()
            cv2.destroyAllWindows()
            test_data = []
            if ret:
                
                tmp = reshape_photo(img)
                if tmp[1] == True:
                    predict_img = reshape_input(img)[0]
                    img = tmp[0]
                else:
                    show_pic = img
                    return render_template('page.html', prediction_display_area='預測失敗，可能是手掌顯示不完全或手臂露出部分太少')
                test_data.append(cv2.resize(cv2.cvtColor(predict_img, cv2.COLOR_BGR2RGB), (50, 50)))
                test_data = np.array(test_data)/255
                output = model.predict(test_data)
                output = np.argmax(output, axis = -1)
                ans = classes[output[0]]
                a, b, c = string_to_point(ans)
                try:
                    show_pic = show_point(img, a, b, c, grid_number = 5, width_rate = 2, whether_grid = False, show_ori = True, whether_show = False, lower_bound = 30, upper_bound = 170)
                    print("predict sucess")
                    return render_template('page.html', prediction_display_area='answer：{}'.format(ans))
                except:
                    show_pic = img
                    return render_template('page.html', prediction_display_area='預測失敗，可能是手掌顯示不完全或手臂露出部分太少')
            else:
                return render_template('page.html', prediction_display_area='尚未重啟相機')
        else: #select image
            release_camera()
            try:
                img = file
            except:
                return render_template('page.html', prediction_display_area='請選擇照片')
            try: #有可能檔案輸入通道只有1, 導致reshape函數失敗
                tmp = reshape_photo(img)
            except:
                show_pic = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                return render_template('page.html', prediction_display_area='預測失敗，可能是手掌顯示不完全或手臂露出部分太少')
            if tmp[1] == True:
                predict_img = reshape_input(img)[0]
                img = tmp[0]
            else:
                show_pic = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                return render_template('page.html', prediction_display_area='預測失敗，可能是手掌顯示不完全或手臂露出部分太少')

            test_data = []
            test_data.append(cv2.resize(predict_img, (50, 50)))
            test_data = np.array(test_data)/255
            output = model.predict(test_data)
            output = np.argmax(output, axis = -1)
            ans = classes[output[0]]
            a, b, c = string_to_point(ans)
            try:
                show_pic = show_point(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), a, b, c, grid_number = 5, width_rate = 2, whether_grid = False, show_ori = True, whether_show = False, lower_bound = 30, upper_bound = 170)
                print("predict sucess")
                return render_template('page.html',prediction_display_area='answer：{}'.format(ans))
            except:
                show_pic = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                return render_template('page.html', prediction_display_area='預測失敗，可能是手掌顯示不完全或手臂露出部分太少')
    return render_template('page.html', prediction_display_area='not post')
####################################################################################################################################################################################################
def init_record():
    global camera_mode, mode, frame, show_pic, select
    camera_mode = False
    mode = 1
    frame = None
    show_pic = None
    select = 2
    release_camera()
@app.route('/record_home')
@login_required
def record_home():
    init_record()
    user = load_user(current_user.id)
    return render_template('record_home.html', show_display_area = '您目前的儲存的位置為: ' + user.user_point)
@app.route("/detect", methods=["POST"])
@login_required
def detect():
    global camera_mode, show_pic, ans
    user = load_user(current_user.id)
    if request.method == "POST":
        if mode == 0: #use camera
            ret, img = camera.read()
            camera_mode = False
            release_camera()
            if ret:
                
                tmp = reshape_photo(img)
                if tmp[1] == True:
                    img = tmp[0]
                else:
                    show_pic = img
                    return render_template('record_home.html', prediction_display_area='沒有偵測到標籤，可能是手掌顯示不完全或手臂露出部分太少', show_display_area = '您目前的儲存的位置為: ' + user.user_point)
                
                
                try:
                    ans = detect_label(img = img, const_rate = 5, whether_show = False, grid_number = 5, width_rate = 2, alter_orix =  0, alter_oriy = 0 , low_bound = 30, upper_bound = 200)
                    a, b, c = string_to_point(ans[0])
                    show_pic = ans[1]
                    for char in ans[0]: #如果有0表示偵測錯誤
                        if char == '0':
                            return render_template('record_home.html', prediction_display_area='沒有偵測到標籤，可能是手掌顯示不完全或手臂露出部分太少', show_display_area = '您目前的儲存的位置為: ' + user.user_point)
                    print("detect sucess")
                    return render_template('record_home.html', prediction_display_area='偵測到的點為：{}'.format(ans[0]), show_display_area = '您目前的儲存的位置為: ' + user.user_point)
                except:
                    show_pic = img
                    return render_template('record_home.html', prediction_display_area='沒有偵測到標籤，可能是手掌顯示不完全或手臂露出部分太少', show_display_area = '您目前的儲存的位置為: ' + user.user_point)
            else:
                return render_template('record_home.html', prediction_display_area='尚未重啟相機', show_display_area = '您目前的儲存的位置為: ' + user.user_point)
        else: #select image
            release_camera()
            try:
                img = file
            except:
                return render_template('record_home.html', prediction_display_area='請選擇照片', show_display_area = '您目前的儲存的位置為: ' + user.user_point)
            try: #有可能檔案輸入通道只有1, 導致reshape函數失敗
                tmp = reshape_photo(img)
            except:
                show_pic = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                return render_template('record_home.html', prediction_display_area='沒有偵測到標籤，可能是手掌顯示不完全或手臂露出部分太少', show_display_area = '您目前的儲存的位置為: ' + user.user_point)
            if tmp[1] == True:
                img = tmp[0]
            else:
                show_pic = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                return render_template('record_home.html', prediction_display_area='沒有偵測到標籤，可能是手掌顯示不完全或手臂露出部分太少', show_display_area = '您目前的儲存的位置為: ' + user.user_point)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            try:
                ans = detect_label(img = img, const_rate = 5, whether_show = False, grid_number = 5, width_rate = 2, alter_orix =  0, alter_oriy = 0 , low_bound = 30, upper_bound = 200)
                a, b, c = string_to_point(ans[0])
                show_pic = ans[1]
                for char in ans[0]: #如果有0表示偵測錯誤
                    if char == '0':
                        return render_template('record_home.html', prediction_display_area='沒有偵測到標籤，可能是手掌顯示不完全或手臂露出部分太少', show_display_area = '您目前的儲存的位置為: ' + user.user_point)
                print("detect sucess")
                return render_template('record_home.html', prediction_display_area='偵測到的點為：{}'.format(ans[0]), show_display_area = '您目前的儲存的位置為: ' + user.user_point)
            except:
                show_pic = img
                return render_template('record_home.html', prediction_display_area='沒有偵測到標籤，可能是手掌顯示不完全或手臂露出部分太少', show_display_area = '您目前的儲存的位置為: ' + user.user_point)
    return render_template('record_home.html', prediction_display_area='not post', show_display_area = '您目前的儲存的位置為: ' + user.user_point)

@app.route("/record_data")
@login_required
def record_data():
    user = load_user(current_user.id)
    user.user_point = str(ans[0])
    db.session.commit()
    print("cur", user.user_point)
    return render_template('record_home.html', show_display_area = '您目前的儲存的位置為: ' + str(ans[0]))
####################################################################################################################################################################################################
def init_pbrecord():
    global camera_mode, mode, frame, show_pic, select
    camera_mode = False
    mode = 1
    frame = None
    show_pic = None
    select = 3
    release_camera()
@app.route('/predictByrecord_home')
@login_required
def predictByrecord_home():
    init_pbrecord()
    user = load_user(current_user.id)
    return render_template('predictByrecord_home.html', show_display_area = '您的儲存的位置為: ' + user.user_point)

@app.route("/show_record", methods=["POST"])
@login_required
def show_record():
    global camera_mode, show_pic
    user = load_user(current_user.id)
    if user.user_point == 'NONE':
        return render_template('predictByrecord_home.html', prediction_display_area='還沒登錄位置', show_display_area = '您的儲存的位置為: ' + user.user_point)
    a, b, c = string_to_point(user.user_point)
    if request.method == "POST":
        if mode == 0: #use camera
            ret, img = camera.read()
            camera_mode = False
            release_camera()
            if ret:
                tmp = reshape_photo(img)
                if tmp[1] == True:
                    img = tmp[0]
                else:
                    show_pic = img
                    return render_template('predictByrecord_home.html', prediction_display_area='顯示失敗，可能是手掌顯示不完全或手臂露出部分太少', show_display_area = '您的儲存的位置為: ' + user.user_point)
                try:
                    show_pic = show_point(img, a, b, c, grid_number = 5, width_rate = 2, whether_grid = False, show_ori = True, whether_show = False, lower_bound = 30, upper_bound = 180)
                    print("show sucess")
                    return render_template('predictByrecord_home.html', prediction_display_area='顯示成功',show_display_area = '您的儲存的位置為: ' + user.user_point)
                except:
                    show_pic = img
                    return render_template('predictByrecord_home.html', prediction_display_area='顯示失敗，可能是手掌顯示不完全或手臂露出部分太少', show_display_area = '您的儲存的位置為: ' + user.user_point)
            else:
                return render_template('predictByrecord_home.html', prediction_display_area='尚未重啟相機', show_display_area = '您的儲存的位置為: ' + user.user_point)
        else: #select image
            release_camera()
            try:
                img = file
            except:
                return render_template('predictByrecord_home.html', prediction_display_area='請選擇照片', show_display_area = '您的儲存的位置為: ' + user.user_point)
            try: #有可能檔案輸入通道只有1, 導致reshape函數失敗
                tmp = reshape_photo(img)
            except:
                show_pic = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                return render_template('predictByrecord_home.html', prediction_display_area='顯示失敗，可能是手掌顯示不完全或手臂露出部分太少', show_display_area = '您的儲存的位置為: ' + user.user_point)
            if tmp[1] == True:
                img = tmp[0]
            else:
                show_pic = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                return render_template('predictByrecord_home.html', prediction_display_area='顯示失敗，可能是手掌顯示不完全或手臂露出部分太少', show_display_area = '您的儲存的位置為: ' + user.user_point)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            try:
                show_pic = show_point(img, a, b, c, grid_number = 5, width_rate = 2, whether_grid = False, show_ori = True, whether_show = False, lower_bound = 30, upper_bound = 180)
                print("show sucess")
                return render_template('predictByrecord_home.html', prediction_display_area='顯示成功', show_display_area = '您的儲存的位置為: ' + user.user_point)
            except:
                show_pic = img
                return render_template('predictByrecord_home.html', prediction_display_area='顯示失敗，可能是手掌顯示不完全或手臂露出部分太少', show_display_area = '您的儲存的位置為: ' + user.user_point)
    return render_template('predictByrecord_home.html', prediction_display_area='not post', show_display_area = '您的儲存的位置為: ' + user.user_point)
####################################################################################################################################################################################################
if __name__ == "__main__":
    load_model('model/model.h5')
    load_data('res/dataset/classes.txt')
    port = 3000
    url = "http://127.0.0.1:{0}".format(port)
    webbrowser.open(url)
    db.create_all()
    app.run(host="0.0.0.0", debug=True, port = port)
    
try:
    camera.release()
    cv2.destroyAllWindows()  
except:
    pass