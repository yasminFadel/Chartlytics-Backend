import urllib
import shutil
import requests
from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import easyocr
import os
import torch
import math
import warnings
import xml.etree.ElementTree as ET
from xml.dom import minidom
import json
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
warnings.filterwarnings("ignore")


reader = easyocr.Reader(['en'])
dir_path = os.path.dirname(os.path.realpath(__file__)) 

legend_weights_path = os.path.join(dir_path,'legend_best.pt')
bar_minor_weights_path = os.path.join(dir_path,'bar_minor.pt')
majorticks_weights_path = os.path.join(dir_path,'major_only.pt')
yolov5_path = os.path.join(dir_path,'yolov5')

legend_model = torch.hub.load(yolov5_path, 'custom',source = 'local',path = legend_weights_path)
bar_minor_model = torch.hub.load(yolov5_path, 'custom',source = 'local', path = bar_minor_weights_path)
major_model = torch.hub.load(yolov5_path, 'custom',source = 'local',path = majorticks_weights_path)
app = Flask(__name__)

################################################## 1- Classification  ##################################################  
  
classification_model_path = os.path.join(dir_path,'LineMobilenet_sigmoid_99_42.h5')
model = load_model(classification_model_path)
def vectorize_image(img, IMG_SIZE = 224):
    img = img.convert('RGB')
    array = image.img_to_array(img)
    array = cv2.resize(array,(IMG_SIZE,IMG_SIZE))
    array = np.expand_dims(array, axis=0) # Adding dimension to convert array into a batch of size (1,299,299,3)
    return array

def process(image_path,IMG_SIZE = 224):
    vectorized_image = vectorize_image(image_path)
    predictions = model.predict(vectorized_image)
    supported = False
    for row in predictions:
        for value in row:
            if value >= 0.6:
                supported = True

    if supported == True:
        prediction_index = [np.argmax(element) for element in predictions]
        if prediction_index[0] == 0:
            return 'Pie Chart'
        elif prediction_index[0] == 1:
            return 'Vertical Bar Chart'
        elif prediction_index[0] == 2:
            return 'Horizontal Bar Chart'
        elif prediction_index[0] == 3:
            return 'Line Plot'
    else:
        return "Class not supported"
    
    

classifyURL = ''
@app.route("/Classify",methods=["GET","POST"])
def startClassify():
    def getURL():
        global classifyURL
        request_data = request.data
        request_data = json.loads(request_data.decode('utf-8'))
        print(request_data)
        classifyURL = request_data['URL']
    getURL()
    global classifyURL
    img = Image.open(requests.get(classifyURL, stream=True).raw)
    return str(process(img))

#################################################   2- Data Extraction    ##################################################
 
################################################# Common Helper Functions ##################################################
# 1- Main Resizing function 
def Resize(image, width, height):
    resized_image = cv2.resize(image,(width, height))
    return resized_image

# 2- Main Sharpening function 
def Sharpening(image):
    sharpen_filter=np.array([[-1,-1,-1],
                [-1,9,-1],
                [-1,-1,-1]])

    sharp_image=cv2.filter2D(image, -1, sharpen_filter) #-1 to get the same depth
    return sharp_image

# 3- Main Noise removal(gridlines) function 
def removeNoise(image):
    dst = cv2.fastNlMeansDenoising(image, None, 8, 7, 21)
    return dst

# 4- Crops Title part of image function 
def crop_title(img):
    # to crop title
    W = img.width
    H = img.height
    cropped_image = np.asarray(img)[:int(H*1/8), :W]
    return cropped_image

# 5- Adjusts bounding boxes to new image size.
def adjust_bboxes(df, width_ratio, height_ratio):
    df['xmin'] = df['xmin'].apply(lambda x: x*width_ratio)
    df['xmax'] = df['xmax'].apply(lambda x: x*width_ratio)
    df['ymin'] = df['ymin'].apply(lambda x: x*height_ratio)
    df['ymax'] = df['ymax'].apply(lambda x: x*height_ratio)
    return df

# 6- Detects texual component of the Bar and Line Chart images using easyOCR. 
def bar_line_ocr(im,thresh,ratio=1,rot=0,margin=0.35):
    # Use ocr to get text like title,legend,legendtitle.
    ocr_pred = reader.readtext(im,min_size=thresh,mag_ratio=ratio,add_margin=margin,batch_size=32,rotation_info = [rot])
    df_label = pd.DataFrame(columns = ['text','xmin','ymin','xmax','ymax'])
    
    for i in range(len(ocr_pred)):
        row = []
        #label text
        row.append(ocr_pred[i][1].strip("'").strip('"'))
        # ocr_pred[2][0] ----> bbox
        boundingbox = ocr_pred[i][0]
        # boundingbox[0] -----> [xmin,ymin] ,   # boundingbox[2] -----> [xmax,ymax]
        row.append(boundingbox[0][0])
        row.append(boundingbox[0][1])
        row.append(boundingbox[2][0])
        row.append(boundingbox[2][1])
        # add new label text.
        df_label.loc[len(df_label)] = row
    return df_label

######################################################  Piecharts helper functions ###################################################

# 1- Crops legend part of image function
def crop_legend(row, image):
    # to squares outside crop image
    Y = int(row['ymin'])
    X = int(row['xmin'])
    H = int(row['ymax'] - row['ymin'])
    W = int(row['xmax'] - row['xmin'])
    image = np.asarray(image).copy()
    image[Y:Y+H, X:X+W] = 255
    return Image.fromarray(image)

# 2- Gets the symbol colors.
def square_color(row, im):
    # to squares outside crop image
    Y = int(row['ymin'])
    X = int(row['xmin'])
    H = int(row['ymax'] - row['ymin'])
    W = int(row['xmax'] - row['xmin'])
    cropped_image = np.asarray(im).copy()[Y:Y+H, X:X+W]
    size = np.shape(cropped_image)

    midX, midY = int(size[1]/2), int(size[0]/2)
    cropped_image = np.asarray(cropped_image)[midY:midY+1, midX:midX+1]

    cropped_image = Image.fromarray(cropped_image)
    return cropped_image.getcolors()[0][1]

# 3- Gets the slice colors and their frequency in respect to the image pixels.
def slice_color_count(im):
    temp = im.getcolors()
    colors = []
    count = []
    for i in range(len(temp)):
        if temp[i][1] >= 250:
            continue
        if temp[i][0] > 600:
            count.append(temp[i][0])
            colors.append(temp[i][1])
    return colors, count

# 4- Linking each slice color with it's percentage of piechart.
def slices_percentage(colors, count):
    colors_percent = {}  # dictionary with color:percentage
    total_pixels = np.sum(np.array(count))
    for i in range(len(count)):
        sliceQuantity = round((count[i] * 100)/total_pixels, 2)
        colors_percent[colors[i]] = sliceQuantity
    return colors_percent

# 5- Linking each slice label with it's percentage of piecharts using colors extracted.
def lbl_percent(df_symbol, df_label, colors_percent):
    label_percentage = {}
    for i in range(len(df_symbol)):
        squareColor = df_symbol["color"].iloc[i]
        label_percentage[df_label['text'].iloc[i]] = colors_percent[squareColor]
    label_percentage = dict(sorted(label_percentage.items(), key=lambda item: item[1], reverse=True))
    return label_percentage

# 6- Removes the false detected symbols using detected symbol colors(ex: watermarks or small letters)
def check_symbols(df_symbol, sliceColors):
    for i in range(len(df_symbol)):
        true_symb = False
        # check for false detected letters
        if df_symbol["color"].iloc[i] == 255 or df_symbol["color"].iloc[i] == 0:
            df_symbol["color"].iloc[i] = np.nan
            continue
        # check for false colored objects
        for j in range(len(sliceColors)):
            if df_symbol['color'].loc[i] == sliceColors[j]:
                true_symb = True
                break
        if true_symb == False:
            df_symbol["color"].iloc[i] = np.nan
    df_symbol = df_symbol.dropna()
    return df_symbol

# 7- Removes false legends using detected symbols(if no symbols inside remove it).
def check_legends(df_legend, df_symbol):
    # if theres an empty legend
    for i in range(len(df_legend)):
        xmin, xmax = df_legend['xmin'].iloc[i], df_legend['xmax'].iloc[i]
        ymin, ymax = df_legend['ymin'].iloc[i], df_legend['ymax'].iloc[i]
        trueLegend = False
        for j in range(len(df_symbol)):
            if (df_symbol['ymin'].iloc[j] > ymin and df_symbol['xmin'].iloc[j] > xmin
            and df_symbol['xmax'].iloc[j] < xmax and df_symbol['ymax'].iloc[j] < ymax):
                trueLegend = True
                break
        if trueLegend == False:
            df_legend['xmax'].iloc[i] = np.nan

    df_legend = df_legend.dropna()
    return df_legend

# 8- Sorts the symbols of horizontal legends using their row and column sizes for further processing.
def sort_df(df, df_type, legendRowSize, legendColSize):
    if legendRowSize == 1:
        df = df.sort_values("xmin").reset_index()
        return df
    df = df.sort_values("ymin").reset_index()
    if df_type == 'label':
        new_df = pd.DataFrame(
            columns=['text', 'xmin', 'ymin', 'xmax', 'ymax'])
        temp_df = pd.DataFrame(
            columns=['text', 'xmin', 'ymin', 'xmax', 'ymax'])
    else:
        new_df = pd.DataFrame(
            columns=['xmin', 'ymin', 'xmax', 'ymax', 'color'])
        temp_df = pd.DataFrame(
            columns=['xmin', 'ymin', 'xmax', 'ymax', 'color'])

    idx = 0
    while idx < len(df):
        df[idx:idx+legendColSize] = df[idx:idx + legendColSize].sort_values('xmin')
        idx += legendColSize
    for i in range(legendColSize):
        j = i
        while j < len(df):
            new_df.loc[len(new_df)] = df.loc[j]
            j += legendColSize
        temp_df.sort_values('ymin')
        new_df = new_df.append(temp_df)
    return new_df

# 9- Extends legend to include all symbols properly.
def extendLegend(df_legend, pie_im, no_of_symbols):
        while (True):
            if df_legend['xmax'].iloc[0] + 5 < pie_im.width:
                df_legend['xmax'].iloc[0] = df_legend['xmax'].iloc[0] + 5

            if df_legend['ymax'].iloc[0] + 5 < pie_im.height:
                df_legend['ymax'].iloc[0] = df_legend['ymax'].iloc[0] + 5

            if df_legend['ymin'].iloc[0] - 5 > 0:
                df_legend['ymin'].iloc[0] = df_legend['ymin'].iloc[0] - 5

            legendBbox = [df_legend['xmin'].iloc[0], df_legend['ymin'].iloc[0],
                df_legend['xmax'].iloc[0], df_legend['ymax'].iloc[0]]
            df_label = pie_ocr(np.array(pie_im),  legendBbox)

            if len(df_label) == no_of_symbols:
                break
            elif len(df_label) > no_of_symbols:
                df_label = pie_join_words(df_label)
        return df_label, df_legend

# 10- Detects texual component of the legend using easyOCR. 
def pie_ocr(im,legendbbox):
    global reader
    # Use ocr to get text like title,legend,legendtitle.
    ocr_pred = reader.readtext(im)
    df_label = pd.DataFrame(columns = ['text','xmin','ymin','xmax','ymax'])
    
    for i in range(len(ocr_pred)):
        row = []
        #label text
        row.append(ocr_pred[i][1])
        # ocr_pred[2][0] ----> bbox
        boundingbox = ocr_pred[i][0]
        # boundingbox[0] -----> [xmin,ymin] ,   # boundingbox[2] -----> [xmax,ymax]
        if boundingbox[0][0] < legendbbox[0] or boundingbox[0][1] < legendbbox[1]  or boundingbox[2][0] > legendbbox[2]  or boundingbox[2][1] > legendbbox[3]:
            continue
        row.append(boundingbox[0][0])
        row.append(boundingbox[0][1])
        row.append(boundingbox[2][0])
        row.append(boundingbox[2][1])
        # add new label text.
        df_label.loc[len(df_label)] = row
    return df_label

# 11- Joins words detected together to form senteces with a certain threshold. 
def pie_join_words(df_label):
    df_label = df_label.sort_values(["ymin"]).reset_index(drop=True)
    for i in range(len(df_label)-1):
        # check range of space between 2 words.
        if abs(df_label['ymin'][i] - df_label['ymin'][i+1]) < 5:
            # get the smaller xmin (first word)
            if df_label['xmin'][i] < df_label['xmin'][i+1]:
                df_label['text'][i] = df_label['text'][i]+' '+df_label['text'][i+1]
            else:
                df_label['text'][i] = df_label['text'][i+1]+' '+df_label['text'][i]
                df_label['xmin'][i],df_label['ymin'][i] = df_label['xmin'][i+1],df_label['ymin'][i+1]   
            df_label['ymax'][i+1] = np.nan
    df_label = df_label.dropna()
    return df_label

######################################   Vertical bar and horizontal bar charts helper functions   ######################################

# 1- Joins words detected together to form senteces with a certain threshold. 
def bar_join_words(df_label,is_xaxis):
    if is_xaxis == True:
        var1= 'ymin'
        var2= 'xmin'
    else:
        var1= 'xmin'
        var2= 'ymin'
    for i in range(len(df_label)-1):
        # check range of space between 2 words.
        if abs(df_label[var1][i] - df_label[var1][i+1]) < 5 and abs(df_label[var2][i] - df_label[var2][i+1]) !=0:
            if df_label['ymax'][i] == -1 :
                continue

            # get the smaller xmin (first word)
            if df_label[var2][i] < df_label[var2][i+1]:
                df_label['text'][i] = df_label['text'][i]+' '+df_label['text'][i+1]
            else:
                df_label['text'][i] = df_label['text'][i+1]+' '+df_label['text'][i]
                df_label[var2][i],df_label[var1][i] = df_label[var2][i+1],df_label[var1][i+1]   
            df_label['ymax'][i]+=df_label['ymax'][i+1]
            df_label['xmax'][i]+=df_label['xmax'][i+1]

            df_label['ymax'][i+1] = -1
    df_label = df_label[(df_label['ymax'] > -1)]
    return df_label

# 2- Crops x_axis part of the image for Vertical Bar charts. 
def crop_xaxis_vertical(im,df):
    Y = int(max(df['ymax']))+10
    cropped_image = np.asarray(im).copy()[Y: , :]
    return cropped_image

# 3- Crops y_axis part of the image for Vertical Bar charts. 
def crop_yaxis_vertical(im,df):
    W = min((df['xmax']))
    H = max(df['ymax'])+40
    cropped_image = np.asarray(im).copy()[:int(H) , :int(W)]
    return cropped_image

# 4- Crops x_axis part of the image for Horizontal Bar charts. 
def crop_xaxis_horizontal(im,df):
    Y = int(max(df['ymax']))
    cropped_image = np.asarray(im).copy()[Y: , :]
    return cropped_image

# 5- Whitnens all image except y_axis part of the image instead of cropping for Horizontal Bar charts (is better for ocr)
def whitenY_hbar(im,df):
    W = int(min(df['xmin']))
    H = max(df['ymax'])
    im = np.asarray(im).copy()
    im[:,int(W):] = 255
    im[int(H):,:] = 255
    return im

# 6- fills the detected minor ticks dataframe with their respective values
def fill_minor(values,add_no):
    for i in range(1,len(values)):
        values[i] = values[i-1] + add_no
    return values

# 7- find minimum difference between two consecutive detected ticks (to avoid ocr problem with small numerical values)
def findmin(y_df_labels):
    minimumdiff = 1000
    val1,val2 = 0, 0 
    for i in range(len(y_df_labels['text'])-1):
        if (y_df_labels['text'][i] - y_df_labels['text'][i+1])<minimumdiff:
            minimumdiff = y_df_labels['text'][i] - y_df_labels['text'][i+1]
            val1 = y_df_labels['text'][i]
            val2 = y_df_labels['text'][i+1]
    return minimumdiff,val1,val2

# 8- fills the detected Bars dataframe with their respective values
def fill_barValue(df_ticks,df_bar,btype):
    for i in df_bar['name'].index:
        closest = 1000
        idx = -1
        if btype == 'v':
            col1 = 'ymin'
            col2 = 'ymax'
        else:
            col1 = 'xmax'
            col2 = 'xmin'
            
        for j in df_ticks['name'].index:
            midpoint = (df_ticks[col1][j] + df_ticks[col2][j])/2
            if abs(df_bar[col1][i] - midpoint) < closest:
                closest  = abs(df_bar[col1][i] - midpoint)
                idx = j
        df_bar['value'][i] =  df_ticks['value'][idx]
    return df_bar


#################################################        Line charts helper functions     ##################################################

# 1- Whitnens all image except x_axis part of the image instead of cropping (is better for ocr)
def whitenXline(im):
    Y =  int(im.height) - int(im.height/8)
    X = int(im.width/8)
    im = np.asarray(im)
    im = im.copy()
    im[:Y+25,:] = 255
    im[:,:X] = 255
    return im

# 2- Whitnens all image except y_axis part of the image instead of cropping (is better for ocr)
def whitenYline(im):
    X = int(im.width/8)
    im = np.asarray(im)
    im = im.copy()
    im[:,X:] = 255
    return im


# 3- Whitnens all components surrounding actual line part of the image.
def crop_line(im,ytitle,xlabel):
    X  = int(im.width/7)#/9
    X2 = int(xlabel)
    Y  =  int(im.height) - int(im.height/8)
    Y2 = int(ytitle)-5
    im = np.asarray(im)
    im = im.copy()
    im[Y: ,:] = 255
    im[:Y2, :] = 255
    
    im[:  ,:X] = 255
    im[: ,X2:] = 255
    return im

# 4- Assigns each detected corner to the closest label bellow it.
def assign_corners(x_df_labels,corners):
    x_df_labels['x'] = (x_df_labels['xmin'] + x_df_labels['xmax'])/2
    x_df_labels['xcorner'] = None
    x_df_labels['ycorner'] = None
    x_df_labels['slope'] = None
    x, y = corners[0].ravel()
    x_df_labels['xcorner'][0] = x
    x_df_labels['ycorner'][0] = y

    for i in range(1,len(x_df_labels['x'])):
        mindistance = 100000
        for j in range(len(corners)):
            x, y = corners[j].ravel()
            if mindistance > abs(x_df_labels['x'][i] - x):
                x_df_labels['xcorner'][i] = x
                x_df_labels['ycorner'][i] = y
                mindistance = abs(x_df_labels['x'][i] - x)
        k = i-1
        while k >= 0:
            if x_df_labels['xcorner'][i] == x_df_labels['xcorner'][k] and x_df_labels['xcorner'][k]!= None:
                if abs(x_df_labels['xcorner'][i] - x_df_labels['x'][i]) < abs(x_df_labels['xcorner'][k] - x_df_labels['x'][k]):
                    x_df_labels['xcorner'][k] = None
                    x_df_labels['ycorner'][k] = None
                else:
                    x_df_labels['xcorner'][i] = None
                    x_df_labels['ycorner'][i] = None
            k = k-1

    lstidx = len(x_df_labels)-1
    x, y = corners[len(corners)-1].ravel()
    x_df_labels['xcorner'][lstidx] = x
    x_df_labels['ycorner'][lstidx] = y
    if  x_df_labels['xcorner'][lstidx] == x_df_labels['xcorner'][lstidx-1]:
        x_df_labels['xcorner'][lstidx-1] = None
        x_df_labels['ycorner'][lstidx-1] = None
    return x_df_labels

# 5- Calculates the slope of each line and assignes it to the label at which the line starts.        
def assign_slope(x_df_labels):
    x_df_labels['angle'] = None
    for i in range(len(x_df_labels)-1):
        if x_df_labels['xcorner'][i] == None:
            continue
        j = i+1
        while j in range(len(x_df_labels)-1):
            if x_df_labels['xcorner'][j] == None:      
                j+=1
            else:
                break
        x_df_labels['slope'][i] = (x_df_labels['ycorner'][i]-x_df_labels['ycorner'][j])/(x_df_labels['xcorner'][i]-x_df_labels['xcorner'][j])
        x_df_labels['angle'][i] = math.degrees(math.atan(x_df_labels['slope'][i]))

    return x_df_labels

# 6- Gets corners of dashed/dotted lines by drawing over it then detecting corners normally.   
def get_dotted_corners(grey_image,croppedLineImg):
    # Apply Canny edge detection to the grayscale image
    edges = cv2.Canny(grey_image, 50, 150, apertureSize=5)

    # Apply Hough Line Transform to detect the lines in the image
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=30, minLineLength=30, maxLineGap=20)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        result = cv2.line(np.asarray(croppedLineImg), (x1, y1), (x2, y2), (0, 0, 0), thickness=1)

    # Apply corner detection
    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # Apply a moving average filter to smooth the line
    kernel_size = 5
    kernel = np.ones((kernel_size,)) / kernel_size
    result = cv2.filter2D(result, -1, kernel)
    
    max_corners = 10
    quality_level = 0.02
    min_distance = 10
    corners = cv2.goodFeaturesToTrack(result, max_corners, quality_level, min_distance)
    corners = sorted(corners, key=lambda c: cv2.boundingRect(c)[0])
    corners = np.int0(corners)
    return corners

############################################   Main function of Charts Data Extraction Sequence   ############################################

# 1- pie data extraction.
def pie_extraction(pie_im,df):
    row = []
    size = 786

    pie_im = pie_im.convert('L')
    # adjust bboxes to orginal image size
    width_ratio,height_ratio = pie_im.width/size, pie_im.height/size
    df = adjust_bboxes(df,width_ratio,height_ratio)

    # splitting classes detected
    df_legend = df.query("name == 'legend'").reset_index(drop=True)
    df_symbol = df.query("name == 'symbol'& confidence > 0.3").reset_index(drop=True)
    df_symbol["color"] = None
    
    sliceColors,sliceCount = slice_color_count(pie_im)
    colors_percent = slices_percentage(sliceColors,sliceCount)
    
    # getting colors of each symbol
    for i in range(len(df_symbol)):
        df_symbol['color'].loc[i] = square_color(df_symbol.iloc[i], pie_im)

    # to eliminate any false symbols then false legends
    df_symbol = check_symbols(df_symbol, sliceColors)
    df_legend = check_legends(df_legend,df_symbol)


    no_of_symbols, no_of_legends = len(df_symbol) , len(df_legend)

    if no_of_legends >= 1 :
        # merging multiple legends into one
        xmin = df_legend.sort_values("xmin")['xmin'].iloc[0]
        ymin = df_legend.sort_values("ymin")['ymin'].iloc[0]
        xmax = df_legend.sort_values("xmax",ascending=False)['xmax'].iloc[0]
        ymax = df_legend.sort_values("ymax",ascending=False)['ymax'].iloc[0]

    elif no_of_legends == 0 :
        # get legend from coordinates of symbol in case of no legend detection
        xmin = df_symbol.sort_values("xmin")['xmin'].iloc[0] 
        ymin = df_symbol.sort_values("ymin")['ymin'].iloc[0] 
        xmax = df_symbol.sort_values("xmax",ascending=False)['xmax'].iloc[0]+ 1/10 *pie_im.width 
        ymax = df_symbol.sort_values("ymax",ascending=False)['ymax'].iloc[0]+ 1/10 *pie_im.height 
        
    df_legend = pd.DataFrame(columns = ['xmin','ymin','xmax','ymax'])
    df_legend.loc[0] = [int(xmin),int(ymin),int(xmax),int(ymax)]
    
    # check if width of the legend is larger than it's heigth then it's a horizontal Legend. 
    horiz_legend = False
    legendColSize = len(df_symbol[abs(df_symbol["ymin"] - df_symbol["ymin"].loc[0])<4])
    legendRowSize = len(df_symbol[abs(df_symbol["xmin"] - df_symbol["xmin"].loc[0])<4])
    if legendColSize > legendRowSize:
        horiz_legend = True

    # Use ocr to get label text and bboxes in the legend region(where no_of_labels = no_of_symbols)
    legendBbox = [df_legend['xmin'].iloc[0],df_legend['ymin'].iloc[0],df_legend['xmax'].iloc[0],df_legend['ymax'].iloc[0]]
    df_label  =  pie_ocr(np.array(pie_im),legendBbox)

    if len(df_label) < no_of_symbols:
        df_label,df_legend = extendLegend(df_legend,pie_im,no_of_symbols)
    elif len(df_label) > no_of_symbols:
        df_label = pie_join_words(df_label).reset_index(drop=True)

    im_no_legend = crop_legend(df_legend.iloc[0],pie_im)
    
    if horiz_legend==True:
        df_label = sort_df(df_label,'label',legendRowSize,legendColSize)
        df_symbol = sort_df(df_symbol,'symbol',legendRowSize,legendColSize)

    else:
        df_label = df_label.sort_values(["ymin"]).reset_index(drop=True)
        df_symbol = df_symbol.sort_values(["ymin"]).reset_index(drop=True)

    label_percentage = lbl_percent(df_symbol,df_label,colors_percent)

    title = ''

    temp_title = reader.readtext(np.asarray(im_no_legend),paragraph = True,detail = 0)

    if len(temp_title)!=0:
        title = temp_title[0]
        
    row.append(title)
    row.append(label_percentage)
    df_label.drop(df_label.index, inplace=True)
    df_symbol.drop(df_symbol.index, inplace=True)
    return row


# 2- Vertical Bar data extraction.
def vbar_extraction(bar_im,bresults,minresult,majresult):
    row = [] 
    size = 768
    bsize = 448
    
    bwidth_ratio,bheight_ratio = bar_im.width/bsize,bar_im.height/bsize
    bar_results = bresults.query("name == 'bars'& confidence > 0.52").reset_index(drop=True)
    
    width_ratio,height_ratio = bar_im.width/size,bar_im.height/size

    minor_results = minresult.query("name == 'minor_ticks' & confidence > 0.6").reset_index(drop=True)
    
    major_results = majresult.query("confidence > 0.1").reset_index(drop=True)

    
    # adjust bboxes to orginal image size
    adjust_bboxes(bar_results,bwidth_ratio,bheight_ratio)
    adjust_bboxes(minor_results,width_ratio,height_ratio)
    adjust_bboxes(major_results,width_ratio,height_ratio)
    minor_results["value"] = None
    
    # x_axis labels and barLabels detection
    bar_im = bar_im.convert('L')
    X_crop = crop_xaxis_vertical(bar_im,bar_results)

    ocr_result = bar_line_ocr(X_crop,30).sort_values('ymin').reset_index(drop=True)
    ocr_result = bar_join_words(ocr_result,True).reset_index(drop=True)

    #get x label then remove it from image.
    x_label = ocr_result.iloc[-1]
    X_crop = X_crop.copy()
    X_crop[int(x_label['ymin']):int(x_label['ymax']),int(x_label['xmin']):int(x_label['xmax'])]=255

    # rotate the image again to get the labels.
    X_crop = Image.fromarray(X_crop).rotate(90,expand=True) #0 lw horizontal 90 lw vertical
    X_crop = np.asarray(X_crop)

    x_df_labels = bar_line_ocr(X_crop,30,2).sort_values('ymin').reset_index(drop=True)
    x_df_labels = bar_join_words(x_df_labels,True)

                                            #######################   Yaxis   ###################################
    
    Y_crop = crop_yaxis_vertical(bar_im,minor_results)
    Y_crop = Image.fromarray(Y_crop).rotate(270,expand=True) #270 to get yaxis label
    Y_crop = np.asarray(Y_crop)
    
    ocr_result = bar_line_ocr(Y_crop,40,2).sort_values('ymin').reset_index(drop=True)
    #get y label then remove it from image.
    y_label = ocr_result.iloc[0]
    Y_crop = Y_crop.copy()
    Y_crop[y_label['ymin']:y_label['ymax']-4,y_label['xmin']:y_label['xmax']]=255  

    # rotate the image again to get the labels.
    no_label_im =Image.fromarray(Y_crop).rotate(90,expand=True) #90 vertical
    
    no_label_im = Resize(np.asarray(no_label_im),no_label_im.width*2,no_label_im.height*2)
    y_df_labels = bar_line_ocr(no_label_im,5,5)#np.asarray(no_label_im)
    y_df_labels['text'] = y_df_labels['text'].apply(lambda x: x.lstrip('.').lstrip('_').lstrip('-').rstrip('.').rstrip('_').rstrip('-'))

    
    #remove non numeric values.
    y_df_labels = y_df_labels[(y_df_labels['text'].str.isnumeric())]
    y_df_labels['text'] = y_df_labels['text'].astype(float)
    
    
    y_df_labels = y_df_labels.sort_values("ymin").reset_index(drop=True)
    major_results = major_results.sort_values("ymin").reset_index(drop=True)
    
    diff_between_majors,num1,num2 = findmin(y_df_labels)

    major_results['value']= None
    major_results['value'][0],major_results['value'][1] = num1,num2
    major_results.drop(['class','confidence'], inplace=True, axis=1)
    major_results['value'].iloc[-1:] = 0.0
    
    minor_results.drop(['class','confidence'], inplace=True, axis=1)
    df_ticks = major_results.append(minor_results).sort_values("ymin").reset_index(drop=True) 
    zero_idx = list(df_ticks['value']).index(0)
    df_ticks = df_ticks.iloc[:zero_idx+1]

    #diff_between_majors = major_results['value'][0] - major_results['value'][1]
    num_minor = list(df_ticks['value']).index(major_results['value'][1]) - list(df_ticks['value']).index(major_results['value'][0])
    add_no = diff_between_majors/num_minor

    # reverse data frame mn zero llakher.
    df_ticks = df_ticks.iloc[::-1].reset_index(drop=True)
    df_ticks['value'] = 0.0
    df_ticks['value'] = fill_minor(df_ticks['value'],add_no)

    #get bar value
    bar_results['value'] = 0.0
    bar_results = bar_results.drop(['class','confidence'], axis=1) .sort_values('ymin').reset_index(drop=True)
    bar_results = fill_barValue(df_ticks,bar_results,'v')


    bar_results = bar_results.sort_values("xmax", ascending=False).reset_index(drop=True)
    if(len(x_df_labels) > len(bar_results)):
        bar_results.loc[len(bar_results)] = [0,0,0,0,'bars',0]
    bar_results['label'] =  x_df_labels['text']


    bar_results['label'] =  bar_results['label'].replace(np.nan, 0)
    bar_results =  bar_results.loc[(bar_results['label']!=0)].reset_index(drop=True)
    bar_results = bar_results.sort_values("xmax").reset_index(drop=True)

    single_dictionary = {}
    for i in range(len(bar_results)):
        single_dictionary[bar_results['label'].iloc[i]] = bar_results['value'].iloc[i]
    
    #title
    cropped_title = crop_title(bar_im)
    title = bar_line_ocr(cropped_title,30)
    title = bar_join_words(title,True).iloc[0]
    row.append(title['text'])
    row.append(x_label['text'])
    row.append(y_label['text'])
    row.append(single_dictionary)
    
    major_results.drop(major_results.index,inplace=True)
    minor_results.drop(minor_results.index,inplace=True)
    bar_results.drop(bar_results.index,inplace=True)
    y_df_labels.drop(y_df_labels.index,inplace=True)
    x_df_labels.drop(x_df_labels.index,inplace=True)
    df_ticks.drop(df_ticks.index,inplace=True)
    return row

# 3- Horizontal Bar data extraction.
def  hbar_extraction(bar_im,bar_results,min_results,maj_results):
    row = []

    #title
    cropped_title = crop_title(bar_im)
    title = bar_line_ocr(cropped_title,30)
    title = bar_join_words(title,True).iloc[0]
    bar_im = np.asarray(bar_im)
    bar_im = bar_im.copy()
    bar_im[int(title['ymin']):int(title['ymax']),int(title['xmin']):int(title['xmax'])]=255
    bar_im = Image.fromarray(bar_im)

    size = 768
    bsize = 448 
    
    bwidth_ratio,bheight_ratio = bar_im.width/bsize,bar_im.height/bsize
    width_ratio,height_ratio = bar_im.width/size,bar_im.height/size

    bar_results = bar_results.query("name == 'bars'& confidence > 0.5").reset_index(drop=True)
    minor_results = min_results.query("name == 'minor_ticks' & confidence > 0.6").reset_index(drop=True)
    major_results = maj_results.query("confidence > 0.8").reset_index(drop=True)
    
    # adjust bboxes to orginal image size
    adjust_bboxes(bar_results,bwidth_ratio,bheight_ratio)
    adjust_bboxes(minor_results,width_ratio,height_ratio)
    adjust_bboxes(major_results,width_ratio,height_ratio)
    minor_results["value"] = None
    
    # x_axis labels and barLabels detection
    bar_im = bar_im.convert('L')
    X_crop = crop_xaxis_horizontal(bar_im,major_results)
    X_crop = Image.fromarray(X_crop)
    X_crop = Resize(np.asarray(X_crop),X_crop.width*2,X_crop.height*2)
    ocr_result =  bar_line_ocr(X_crop,40).sort_values('ymin').reset_index(drop=True)
    ocr_result = bar_join_words(ocr_result,True).reset_index(drop=True)

    # get x label then remove it from image.
    x_label = ocr_result.iloc[-1]
    X_crop = X_crop.copy()
    X_crop[int(x_label['ymin']):int(x_label['ymax']),int(x_label['xmin']):int(x_label['xmax'])]=255

    # rotate the image again to get the labels.
    x_df_labels =  bar_line_ocr(X_crop,10,4).sort_values('xmin',ascending = False).reset_index(drop=True)
    if len(x_df_labels)<=1:
        X_crop = Sharpening(X_crop)
        x_df_labels = bar_line_ocr(X_crop,10,4).sort_values('xmin',ascending = False).reset_index(drop=True)
    
                                    #######################   Yaxis   ###################################
    
    Y_crop = whitenY_hbar(bar_im,minor_results)
    Y_crop = Image.fromarray(Y_crop).rotate(270,expand=True) #270 to get yaxis label
    Y_crop = np.asarray(Y_crop)
    Y_crop = removeNoise(Y_crop)
    

    ocr_result = bar_line_ocr(Y_crop,40,2).sort_values('ymin').reset_index(drop=True)

    #get y label then remove it from image.
    y_label = ocr_result.iloc[0]
    Y_crop = Y_crop.copy()
    Y_crop[int(y_label['ymin']):int(y_label['ymax']-4),int(y_label['xmin']):int(y_label['xmax'])]=255  

    # rotate the image again to get the labels.
    no_label_im = Image.fromarray(Y_crop).rotate(90,expand=True) #90 vertical
    y_df_labels = bar_line_ocr(np.asarray(no_label_im),20)
    y_df_labels = bar_join_words(y_df_labels,True)
    y_df_labels = y_df_labels.sort_values('ymin').reset_index(drop=True)
    y_df_labels['text'] = y_df_labels['text'].apply(lambda x: x.rstrip("_").rstrip("-").lstrip('~'))

    #remove non numeric values.
    x_df_labels = x_df_labels[(x_df_labels['text'].str.isnumeric())]
    x_df_labels['text'] = x_df_labels['text'].astype(float)
    x_df_labels = x_df_labels.sort_values("xmin",ascending=False).reset_index(drop=True)
    major_results = major_results.sort_values("xmin",ascending=False).reset_index(drop=True)

    major_results['value']= None
    major_results['value'][0],major_results['value'][1] = x_df_labels['text'][0],x_df_labels['text'][1]
    major_results.drop(['class','confidence'], inplace=True, axis=1)
    major_results['value'].iloc[-1:] = 0.0
    
    minor_results.drop(['class','confidence'], inplace=True, axis=1)
    df_ticks = major_results.append(minor_results).sort_values("xmin",ascending=False).reset_index(drop=True) 
    zero_idx = list(df_ticks['value']).index(0)
    df_ticks = df_ticks.iloc[:zero_idx+1]
    

    diff_between_majors = major_results['value'][0] - major_results['value'][1]
    num_minor = list(df_ticks['value']).index(major_results['value'][1]) - list(df_ticks['value']).index(major_results['value'][0])
    add_no = diff_between_majors/num_minor

    # reverse data frame mn zero llakher.
    df_ticks = df_ticks.iloc[::-1].reset_index(drop=True)
    df_ticks['value'] = 0.0
    df_ticks['value'] = fill_minor(df_ticks['value'],add_no)

    #get bar value
    bar_results['value'] = 0.0
    bar_results = bar_results.drop(['class','confidence'], axis=1).sort_values('xmax',ascending = False).reset_index(drop=True)
    df_ticks = fill_barValue(df_ticks,bar_results,'h')

    bar_results = bar_results.sort_values("ymin").reset_index(drop=True)
    if(len(y_df_labels) > len(bar_results)):
        bar_results.loc[len(bar_results)] = [0,0,0,0,'bars',0]
    bar_results['label'] =  y_df_labels['text'] 
    
    #creating dictionary for the template
    single_dictionary = {}
    bar_results['label'] =  bar_results['label'].replace(np.nan, 0)
    bar_results =  bar_results.loc[(bar_results['label']!=0)].reset_index(drop=True)
    for i in range(len(bar_results)):
        single_dictionary[bar_results['label'].iloc[i]] = bar_results['value'].iloc[i]

    row.append(title['text'])
    row.append(x_label['text'])
    row.append(y_label['text'])
    row.append(single_dictionary)
    
    major_results.drop(major_results.index,inplace=True)
    minor_results.drop(minor_results.index,inplace=True)
    bar_results.drop(bar_results.index,inplace=True)
    y_df_labels.drop(y_df_labels.index,inplace=True)
    x_df_labels.drop(x_df_labels.index,inplace=True)
    df_ticks.drop(df_ticks.index,inplace=True)
    return row

    
# 4- Line data extraction.
def line_extraction(line_im):
    row = [] 
    size = 768   
    line_im = Image.fromarray(removeNoise(np.asarray(line_im)))  
    # x_axis labels
    line_im = line_im.convert('L')

                    ##############     xaxis       #################

    whitenedim = whitenXline(line_im)
    ocr_result = bar_line_ocr(whitenedim,30).sort_values('ymin').reset_index(drop=True)

    # get x label then remove it from image.
    x_label = ocr_result.iloc[-1]
    
    whitenedim = whitenedim.copy()
    whitenedim[int(x_label['ymin']+5):int(x_label['ymax']),int(x_label['xmin']):int(x_label['xmax'])]=255
    whitenedim = np.asarray(whitenedim)
    whitenedim = removeNoise(whitenedim)
    


    x_df_labels = ocr_result.iloc[:-1]
    x_df_labels = x_df_labels.sort_values('xmin').reset_index(drop=True)
    x_df_labels['text'] = x_df_labels['text'].apply(lambda x: x.strip("  _~,"))


                    ##############     yaxis       #################

    Y_crop = whitenYline(line_im)
    ocr_result = bar_line_ocr(Y_crop,20).sort_values('ymin').reset_index(drop=True)
    
    Y_crop = Image.fromarray(Y_crop).rotate(270,expand=True) #270 to get yaxis label
    Y_crop = Resize(np.asarray(Y_crop),Y_crop.width*2,Y_crop.height*2)
    ocr_result = bar_line_ocr(Y_crop,30).sort_values('ymin').reset_index(drop=True)

    # get y label then remove it from image.
    y_label = ocr_result.iloc[0]
    y_label['text'] = y_label['text'].strip("~,.")


                    ##################   title    ######################
    
    #title
    cropped_title = crop_title(line_im)
    title = bar_line_ocr(cropped_title,30).iloc[0]

                    ################ rest of data extraction ##################
    
    line_im = Image.fromarray(removeNoise(np.asarray(line_im)))  
    line_im = line_im.convert('RGB')
    croppedLineImg = crop_line(line_im,title['ymax'],x_df_labels['xmax'][len(x_df_labels)-1])

    grey_image = cv2.cvtColor(croppedLineImg, cv2.COLOR_BGR2GRAY)
    max_corners = 10
    quality_level = 0.02
    min_distance = 10

    # Apply corner detection
    corners = cv2.goodFeaturesToTrack(grey_image, max_corners, quality_level, min_distance)
    corners = sorted(corners, key=lambda c: cv2.boundingRect(c)[0])
    corners = np.int0(corners)
    
    #check if dotted
    if(len(corners) == max_corners):
        corners = get_dotted_corners(grey_image,croppedLineImg)
    
    # assign corners and slope
    x_df_labels = assign_corners(x_df_labels,corners)
    x_df_labels = assign_slope(x_df_labels)
    row.append(title['text'].lstrip('~'))
    row.append(x_label['text'].lstrip('~'))
    row.append(x_df_labels.copy())
    row.append(y_label['text'].lstrip('~'))
    x_df_labels.drop(x_df_labels.index,inplace=True)
    return row  

#########################################     Summary Generation Templates    ######################################### 

# 1- Pie Charts Template
def generate_pieText(pie_info):
    output = ''
    output += 'This is a PieChart representing the percentages of ' + pie_info[0] + '\n'
    labels = list(pie_info[1].keys())
    percents = list(pie_info[1].values())
    if len(labels) == 1:
        output +=  labels[0] + ' is the only one with ' + str(percents[0]) +'%\n\n'

    output +=  labels[0] + ' is the highest with percentage ' + str(percents[0]) +'%\n'
    for j in range(1,len(labels)-1):
        output +=  'followed by ' + labels[j] + ' with ' + str(percents[j]) + '%\n'
    output += 'Lastly is ' + labels[-1] + ' with the least percentage with ' + str(percents[-1]) + '%\n'
    output += '\n\n'
    return output


# 2- V_BAR and H_BAR Charts Templates
def generate_barText(bar_info):
    output = ""
    # keys = []
    # values = []
    output += 'This Bar Chart is representing '+ bar_info[0].lstrip(" ").rstrip(" ").lstrip("~").rstrip("~") + "\n"
    output += "The y-axis represents " + bar_info[2].lstrip(" ").rstrip(" ").lstrip("~").rstrip("~") + ', while the x-axis represents ' \
    + bar_info[1].lstrip(" ").rstrip(" ").lstrip("~").rstrip("~") + "\n"
    
    dictionary = bar_info[3]
    
    interchanged_dict = {}
    for barValue, value in dictionary.items():
        if value in interchanged_dict:
            interchanged_dict[value].append(barValue)
        else:
            interchanged_dict[value] = [barValue]
    interchanged_dict = dict(sorted(interchanged_dict.items(), reverse=True))

    keys = list(interchanged_dict.keys())
    values = list(interchanged_dict.values())
    min_value = min(keys)
    max_value = max(keys)
                
    # for maximum value items
    listofmax = interchanged_dict[max_value]
    for i in range(len(listofmax)):
        output += str(listofmax[i]).lstrip(" ").rstrip(" ")
        if i == len(listofmax)-2:
            output += ' and '
        elif i != len(listofmax)-1:
            output += ', '
        
    if len(listofmax) == 1:
        output+= ' is '
    else:
        output+= ' are '
    output += 'the maximum with value ' + str(max_value) + "\n" 

            
    # for items in between
    for barValue in interchanged_dict.keys():
        if barValue == max_value or barValue == min_value:
            continue
        listofkey = interchanged_dict[barValue]
        output += "followed by " 
        for i in range(len(listofkey)):
            output += str(listofkey[i]).lstrip(" ").rstrip(" ")
            if i == len(listofkey)-2:
                output += ' and '
            elif i != len(listofkey)-1:
                output += ', '
        output += ' with value ' + str(barValue) + '\n'

    # for least value items
    listofmin = interchanged_dict[min_value]
    output += 'lastly ' 
    for i in range(len(listofmin)):
        output += str(listofmin[i]).lstrip(" ").rstrip(" ")
        if i == len(listofmin)-2:
            output += ' and '
        elif i != len(listofmin)-1:
            output += ', '

    output += ' with value ' + str(min_value) + '\n'

    return output

# 3- Line Charts Template
def generate_lineText(line_info):
    output = ""
    keys = []
    values = []
    output += 'This is a Line Chart of '+ str(line_info[0]) +' representing the visualization between '
    output += str(line_info[1]) + ' and ' + str(line_info[3])
    x_labels = line_info[2]
    x_values = list(x_labels['text'])
    xmin_value = min(x_values)
    xmax_value = max(x_values)

    df = pd.DataFrame(columns = ['state','year1','year2']) 
    output += '\n'+ 'The values of x axis are ranged between '
    output += str(xmin_value) + ' and ' + str(xmax_value) + '\n'
    output +='This chart ' 
    # if slope is not none
    change = ""
    degree = ""
    i = 0
    while  i in range(len(x_labels)-1):
        row = []
        if x_labels['slope'][i] < 0:
            change = "increases"
        elif x_labels['slope'][i] > 0:
            change = "decreases"
        else:
            change = "almost constant"
            
        if change == "almost constant":        
            degree = " "
        else:   
            if abs(x_labels['slope'][i]) < 0.3 :
                    degree = " slightly"
            elif abs(x_labels['slope'][i]) >= 0.3 :
                    degree = " sharply"

        row.append(change+degree)
        
        j = i+1
        while j in range(len(x_labels)-1):
            if x_labels['angle'][j] == None:      
                j+=1
            else:
                break
        row.append(str(x_values[i]))
        row.append(str(x_values[j]))
        df.loc[len(df)] = row
        i = j


    i=0
    while  i in range(len(df['state'])):
        output += df['state'][i] + ' from ' + str(df['year1'][i])
        j = i+1
        while j in range(len(df['state'])-1):
            if df['state'][j] == df['state'][i]:      
                j+=1
            else:
                break
        if j > i+1:
            output += " till "+ str(df['year2'][j-1]) 
            i = j

        else:
            output += " till "+ str(df['year2'][i]) 
            i = i+1
            
        if i < (len(df['state'])):
                output += '\nthen it '
        
    output+='\n\n'
        
    return output 
# Connects with flutter.
type = -1
URL = ''
@app.route("/ExtractData",methods=["GET","POST"])
def start():
    def getURL():
        global type,URL
        request_data = request.data
        request_data = json.loads(request_data.decode('utf-8'))
        URL = request_data['URL']
        type = request_data['type']
    try:
        getURL()
        global type,URL
        img = Image.open(requests.get(URL, stream = True).raw)
        if  type == 0:
            size = 786
            img = img.convert('L').resize((size,size))
        
            results = legend_model(img).pandas().xyxy[0]

            info = pie_extraction(img, results)
            final = generate_pieText(info)

        elif type == 3:
            img = np.asarray(img)
            info = line_extraction(img)
            final = generate_lineText(info) 

        else:
            size = 768
            bsize = 448
            img = Image.fromarray(removeNoise(np.asarray(img)))

            bar_results = bar_minor_model(img.resize((bsize,bsize))).pandas().xyxy[0]
            min_results = bar_minor_model(img.resize((size,size))).pandas().xyxy[0]
            maj_results = major_model(img.resize((size,size))).pandas().xyxy[0]

            if  type == 1:
                info = vbar_extraction(img,bar_results,min_results,maj_results)
            elif type == 2:
                info = hbar_extraction(img,bar_results,min_results,maj_results)

            final = generate_barText(info)
    except:
        final =  "Sorry :(  Could not Summarise your Chart Image"
    return str(final)



if __name__ == '__main__':
    app.run(host = 'localhost')