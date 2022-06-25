import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from tkinter import filedialog as fd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
from PIL import Image, ImageTk
from keras.models import load_model
import numpy as np


def on_closing():
    if messagebox.askokcancel("Выход из приложения", "Хотите выйти из приложения?"):
        window.destroy()


def neron_func():

    def start_ner():
        # подгрузка изображения
        file_name = fd.askopenfile()
        image = Image.open(file_name.name)
        image = image.resize((300, 300))
        image = ImageTk.PhotoImage(image)
        label_img['image'] = image
        label_img.image = image

        # Работа НС
        # CATEGORIES = ['actinic keratosis',
        #               'basal cell carcinoma',
        #               'dermatofibroma',
        #               'melanoma',
        #               'nevus',
        #               'pigmented benign keratosis',
        #               'seborrheic keratosis',
        #               'squamous cell carcinoma',
        #               'vascular lesion']
        CATEGORIES = ['akies',
                      'df',
                      'mel',
                      'bcc',
                      'vasc',
                      'nv',
                      'bkl']
        img_array = cv2.imread(file_name.name, cv2.IMREAD_COLOR)
        IMG_SIZE = 100
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        new_array = np.array(new_array)
        x = np.expand_dims(new_array, axis=0)
        res = model.predict(x)

        index = np.argmax(res)
        labels['text'] = 'Предполагается: '+ CATEGORIES[index]


    ner_window = tk.Toplevel(window)
    ner_window.title("Окно нейронной сети")
    ner_window.geometry("350x400+750+200")  # размеры окна +позиция окна на экране
    ner_window.wm_attributes("-topmost", 1)
    ner_window.resizable(False, False)

    frame_img = tk.Frame(ner_window, width=350, height=340)
    frame_panel = tk.Frame(ner_window, width=350, height=20)

    frame_img.grid(row=0, column=0)
    frame_panel.grid(row=1, column=0)


    label_img = tk.Label(frame_img)
    label_img.place(x=20, y=20)
    start_button = tk.Button(frame_panel,
                             text='Загрузка изображения',
                             command=start_ner)
    labels = tk.Label(frame_panel, text='')

    start_button.pack(padx=10, pady=2)
    labels.pack(padx=10, pady=2)
    model = load_model('my_model.h5')
    print(model.summary())
    ner_window.mainloop()


def patient_insert():
    print('добавление пациента')


def patient_delete():
    print('удаление пациента')


def patient_update():
    print('изменение пациента')


window = tk.Tk()
window.protocol("WM_DELETE_WINDOW", on_closing)
# настройка самого окна
icon = tk.PhotoImage(file='snake.png')  # иконка
window.title("Систаема обнаружения")
window.geometry("+200+200")  # размеры окна +позиция окна на экране
window.wm_attributes("-topmost", 1) # окно будет находится поверх других окон
window.iconphoto(False, icon)   # загрузка иконки в окно
window.resizable(False, False)  # изменение размера окна

frame_list = tk.Frame(window, width=650, height=400)
frame_status = tk.Frame(window, width=140, height=400, bg='gray')

frame_list.grid(row=0, column=0, sticky='sn')   # columnspan - этим параметром можно выделить виджет на несколько колонок таблицы
frame_status.grid(row=0, column=1, sticky='ns')  # sticky - можно растянуть виджет по сторонам света

# создание кнопок в правой части приложения
patient_ins = tk.Button(frame_status,
                 text='Добавить пациента',
                 command=patient_insert,
                 state=tk.DISABLED)
patient_upd = tk.Button(frame_status,
                 text='Изменить пациента',
                 command=patient_delete,
                 state=tk.DISABLED)
patient_del = tk.Button(frame_status,
                 text='Удалить пациента',
                 command=patient_update,
                 state=tk.DISABLED)
neron_start = tk.Button(frame_status,
                 text='Запуск нейросети',
                 command=neron_func)

lst = [
    ( 'Иван Иванович Иванов', 28, 'м'),
    ( 'Василий Васильевич Васильев', 54, 'м'),
    ( 'Маркфья Альбертовна Бердянск', 44, 'ж'),
    ( 'Иван Иванович Иванов', 28, 'м'),
    ( 'Василий Васильевич Васильев', 54, 'м'),
    ('Маркфья Альбертовна Бердянск', 44, 'ж'),
    ( 'Иван Иванович Иванов', 28, 'м'),
    ( 'Василий Васильевич Васильев', 54, 'м'),
    ( 'Маркфья Альбертовна Бердянск', 44, 'ж'),
    ( 'Василий Васильевич Васильев', 54, 'м'),
    ( 'Маркфья Альбертовна Бердянск', 44, 'ж'),
    ( 'Иван Иванович Иванов', 28, 'м'),
    ( 'Василий Васильевич Васильев', 54, 'м'),
    ( 'Маркфья Альбертовна Бердянск', 44, 'ж'),
    ( 'Иван Иванович Иванов', 28, 'м'),
    ( 'Василий Васильевич Васильев', 54, 'м')
]
# создание вывода таблицы
table = ttk.Treeview(frame_list, show='headings')   # show позволяет удалить первый пустой столбец (корневой элемент)
heads = ['Имя', 'Возраст', 'Пол']
table['columns'] = heads
table['displaycolumns'] = [ 'Возраст', 'Имя', 'Пол'] # изменение порядка столбцов

for header in heads:
    table.heading(header, text=header, anchor='center')
    table.column(header, anchor='center')
for row in lst:
    table.insert('', tk.END, values=row) # (идентификатор родителя(его тут нет), параметр_что нужно следующую строку
    # перенести на другую строку, что нужно передавать)

scroll_pane = ttk.Scrollbar(frame_list, command=table.yview)    # добавление скроллинга
table.configure(yscrollcommand=scroll_pane.set)  # config добавление атрибута в table,
                                                # ysscrollcom-связывание скролла с таблицей


scroll_pane.pack(side=tk.RIGHT, fill=tk.Y)
table.pack(expand=tk.YES, fill=tk.BOTH)
patient_ins.pack(expand=True, padx=20, pady=0)
patient_upd.pack(expand=True, padx=20, pady=0)
patient_del.pack(expand=True, padx=20, pady=0)
neron_start.pack(expand=True, padx=20, pady=0)
window.mainloop()