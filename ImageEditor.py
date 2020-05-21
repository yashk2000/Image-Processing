import PySimpleGUI as sg
import cv2
import numpy as np

def adjust_gamma(image, gamma=5.0):
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	return cv2.LUT(image, table)

def adjust_contrast(img, value):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=value, tileGridSize=(8,8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl,a,b))

    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def main():
    sg.theme('LightGrey')

    layout = [
            [sg.Text('Video Input', size=(60, 1), justification='center')],
            [sg.Image(filename='', key='IMAGE')],
            [sg.Text('Brightness', size=(15, 1), auto_size_text=False, justification='left'),
            sg.Slider((1,100),
                       disable_number_display=True, 
                       enable_events=True,
                       size=(50, 20),
                       orientation='h',
                       key='BRIGHTNESS',
                       default_value=10)],
            [sg.Text('Contrast', size=(15, 1), auto_size_text=False, justification='left'),
            sg.Slider((1,100),
                       disable_number_display=True, 
                       enable_events=True,
                       size=(50, 20),
                       orientation='h',
                       key='CONTRAST',
                       default_value=5)],
         ]
    
    window = sg.Window('Feed from webcam', layout, location=(800, 400))

    cap = cv2.VideoCapture(0)

    while True:
        event, values = window.read(timeout=20)
        if event == 'Exit' or event == sg.WIN_CLOSED:
            break

        ret, frame = cap.read()

        if (frame.any() != None):
            value_b = values['BRIGHTNESS']
            value_b *= 0.1
            if (value_b != 0):
                frame = adjust_gamma(frame, gamma=value_b)
            
            value_c = values['CONTRAST']
            value_c *= 0.1
            frame = adjust_contrast(frame, value_c)

        imgbytes = cv2.imencode('.png', frame)[1].tobytes()
        window['IMAGE'].update(data=imgbytes)

    window.close()


main()
