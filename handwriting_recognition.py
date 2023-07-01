import sys
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap, QImage
import sys
from PyQt5 import QtCore, QtGui, QtWidgets, uic
import numpy as np
import cv2 
from keras.models import load_model
import imutils
from sklearn.preprocessing import LabelBinarizer

model = load_model('handwriting_recognition.h5')

#Kare içine alınan karakterlerin x kordinatlarına göre soldan sağa sıralanması
def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    key=lambda b:b[1][i], reverse=reverse))
    return (cnts, boundingBoxes)

# Bir noktanın x kordinatlarını döndüren fonksiyon
def get_x(point):
    return point[0][0]

# Bir noktanın y kordinatlarını döndüren fonksiyon
def get_y(point):
    return point[0][1]


# Karakterleri algılayan, tahmin eden, kelime ve cümle haline getiren ana fonksiyon
def get_letters(img):
    # Modelin yapacağı tahminlerin ikili sistemden geri dönüştürülmesi için ayarlanan transform listesi
    LB = LabelBinarizer()
    trans = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    trans = LB.fit_transform(trans)

    # Görüntünün okunması ve gri ton haline getirilmesi
    letters = []
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret,thresh1 = cv2.threshold(gray ,127,255,cv2.THRESH_BINARY_INV)
    dilated = cv2.dilate(thresh1, None, iterations=4)

    # Görüntüdeki karakterlerin algılanması ve kareler halinde depolanması
    cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts)[0]

    # Her bir karenin en düşük x ve y değerlerini bulan fonksiyon
    min_x_coords = [min(contour, key=get_x) for contour in cnts]
    min_y_coords = [min(contour, key=get_y) for contour in cnts]

    # Her bir karenin en yüksek x ve y değerlerini bulan fonksiyon
    max_x_coords = [max(contour, key=get_x) for contour in cnts]
    max_y_coords = [max(contour, key=get_y) for contour in cnts]

    min_Xs = []
    min_Ys = []
    max_Xs = []
    max_Ys = []

    # Her bir karenin minimum ve maximum x ve y değerlerini, karenin sıra sayısıyla birlikte kaydedilmesi
    for i in range(len(cnts)):
        x, y = min_x_coords[i][0]
        min_Xs.append([x, i])
        x, y = min_y_coords[i][0]
        min_Ys.append([y, i])

    for i in range(len(cnts)):
        x, y = max_x_coords[i][0]
        max_Xs.append([x, i])
        x, y = max_y_coords[i][0]
        max_Ys.append([y, i])

    min_Xs_copy = min_Xs.copy()
    max_Xs_copy = max_Xs.copy()
    min_Ys_sorted = sorted(min_Ys)
    words = []
    word = []
    final_sentence = []

    # Görüntüdeki karakterlerin Satır satır sıralanması, tahmin edilmesi ve cümle olarak birleştirilmesi
    sorting = True
    while sorting:
        # Görüntüdeki karakterleri en düşük y değerine satırlaştırılması (en yukarıdan en aşağıya)
        count = 0
        max_y = max_Ys[min_Ys_sorted[0][1]][0]
        min_y = min_Ys[min_Ys_sorted[0][1]][0]
        extend = int((max_Ys[min_Ys_sorted[0][1]][0] - min_Ys[min_Ys_sorted[0][1]][0]) / 2)
        for i in range(len(min_Ys_sorted)):
            temp = int((max_Ys[min_Ys_sorted[i][1]][0] + min_Ys[min_Ys_sorted[i][1]][0]) / 2)
            if temp < max_y + extend and temp >= min_y:
                words.append(min_Ys_sorted[i][1])
                min_Ys_sorted.remove(min_Ys_sorted[i])
                min_Ys_sorted.insert(i, "a")
                count += 1
        for i in range(count):
            min_Ys_sorted.remove("a")
        
        # Satırlaştırılmış karakterlerin en küçük x değerlerine göre sıralanması ve aralarındaki boşlukların uzaklıklarına göre kelimelere ayrılması
        for i in words:
            word.append(max_Xs[i])
            word_sorted = sorted(word)
        final_word = []
        for i in range(len(word_sorted)):
            final_word.append(word_sorted[i][1])
        g = 0
        final_word_split = []
        if len(final_word) > 1:
            tempXs = min_Xs_copy[final_word[1]][0]
            tempXs2 = max_Xs_copy[final_word[0]][0]
            temp2 = tempXs - tempXs2
        for i in final_word:
            g += 1
            if g < len(final_word):
                temp3 = min_Xs[final_word[g]][0] - max_Xs[i][0]

            if temp3 > temp2 + temp2:
                final_word_split = final_word[g:]
                for i in range(len(final_word_split)):
                    final_word.remove(final_word_split[i])
            temp2 = temp3
        
        # Satırlardaki karakterlerin tahminlerinin yapılması ve kelimeler haline getirilmesi
        for i in final_word:
            c = cnts[i]
            if cv2.contourArea(c) > 10:
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi = gray[y:y + h, x:x + w]
            thresh = cv2.threshold(roi, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            thresh = cv2.resize(thresh, (32, 32), interpolation = cv2.INTER_CUBIC)
            thresh = thresh.astype("float32") / 255.0
            thresh = np.expand_dims(thresh, axis=-1)
            thresh = thresh.reshape(1,32,32,1)
            ypred = model.predict(thresh)
            ypred = LB.inverse_transform(ypred)
            [x] = ypred
            letters.append(x)
        pred_word = get_word(letters)
        final_sentence.append(pred_word)
        pred_sentence = sentence(final_sentence)
        letters.clear()
        word.clear()
        words.clear()
        word_sorted.clear()
        final_word.clear()

        # Eğer satırda birden fazla kelime varsa diğer kelimelerin de tahmin edilmesi
        if len(final_word_split) > 0:
            sorting2 = True
            while sorting2:
                g = 0
                for i in range(len(final_word_split)):
                    final_word.append(final_word_split[i])
                final_word_split.clear()
                if len(final_word) > 1:
                    tempXs = min_Xs_copy[final_word[1]][0]
                    tempXs2 = max_Xs_copy[final_word[0]][0]
                    temp2 = tempXs - tempXs2
                elif len(final_word) == 1:
                    temp2 = 0
                for i in final_word:
                    g += 1
                    if g < len(final_word):
                        temp3 = min_Xs_copy[final_word[g]][0] - max_Xs_copy[i][0]
                    if temp3 > temp2 + temp2:
                        final_word_split = final_word[g:]
                        for i in range(len(final_word_split)):
                            final_word.remove(final_word_split[i])
                    temp2 = temp3
                for i in final_word:
                    c = cnts[i]
                    if cv2.contourArea(c) > 10:
                        (x, y, w, h) = cv2.boundingRect(c)
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    roi = gray[y:y + h, x:x + w]
                    thresh = cv2.threshold(roi, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                    thresh = cv2.resize(thresh, (32, 32), interpolation = cv2.INTER_CUBIC)
                    thresh = thresh.astype("float32") / 255.0
                    thresh = np.expand_dims(thresh, axis=-1)
                    thresh = thresh.reshape(1,32,32,1)
                    ypred = model.predict(thresh)
                    ypred = LB.inverse_transform(ypred)
                    [x] = ypred
                    letters.append(x)
                pred_word = get_word(letters)
                final_sentence.append(pred_word)
                letters.clear()
                final_word.clear()
                if len(final_word_split) == 0:
                    sorting2 = False
            pred_sentence = sentence(final_sentence)
        if len(min_Ys_sorted) == 0:
            sorting = False
    return pred_sentence, image

# Tahmin edilen karakterlerin kelime haline getirilmesi
def get_word(letter):
    word = "".join(letter)
    return word
# Tahmin edilen kelimelerin cümle haline getirilmesi
def sentence(word):
    sentence = " ".join(word)
    return sentence

# Kullanıcı arayüzü
class MainWidget(QtWidgets.QWidget):

    def __init__(self):
        super(MainWidget, self).__init__()

        self.pen_color = None
        self.button_save = None
        self.button_load = None
        self.prediction = None
        self.last_y = None
        self.last_x = None
        self.label = None
        self.container = None

        self.initUI()

    def initUI(self):
        self.container = QtWidgets.QVBoxLayout()
        self.container.setContentsMargins(0, 0, 0, 0)

        self.label = QtWidgets.QLabel()
        canvas = QtGui.QPixmap(1200, 300)
        canvas.fill(QtGui.QColor("black"))
        self.label.setPixmap(canvas)
        self.last_x, self.last_y = None, None

        self.prediction = QtWidgets.QLabel('Prediction: ...')
        self.prediction.setFont(QtGui.QFont('Monospace', 20))

        self.button_load = QtWidgets.QPushButton('LOAD IMAGE')
        self.button_load.clicked.connect(self.clicker)

        self.button_save = QtWidgets.QPushButton('PREDICT')
        self.button_save.clicked.connect(self.predict)

        self.container.addWidget(self.label)
        self.container.addWidget(self.prediction, alignment=QtCore.Qt.AlignHCenter)
        self.container.addWidget(self.button_load)
        self.container.addWidget(self.button_save)

        self.setLayout(self.container)

    def clicker(self):
        self.fname = QFileDialog.getOpenFileName(self, "Open File", "c:\\gui\\", "All Files (*);;PNG Files (*.png);;Jpg Files (*.jpg)")
        if self.fname:
            self.pixmap = QPixmap(self.fname[0])
            self.label.setPixmap(self.pixmap)

    def predict(self):
        letter, image = get_letters(self.fname[0])
        self.prediction.setText('Prediction: ' + letter)
        height, width, channel = image.shape
        bytesPerLine = 3 * width
        qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap(qImg))

            

        

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.mainWidget = None
        self.initUI()

    def initUI(self):
        self.mainWidget = MainWidget()
        self.setCentralWidget(self.mainWidget)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    mainApp = MainWindow()
    mainApp.setWindowTitle('HANDWRITING RECOGNITION')
    mainApp.show()
    sys.exit(app.exec_())
