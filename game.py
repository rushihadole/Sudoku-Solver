import cv2
import numpy as np
import tensorflow as tf
import sudoku

###################################
path = "res/sudoku.jpg"
width = 450
height = 450
img = cv2.imread(path)
img = cv2.resize(img, (width, height))
blank_img = np.zeros((height,width,3),np.uint8)
model = tf.keras.models.load_model("myModel.h5")
#################################

# FUNCTIONS
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def preprocess(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5,5), 1)
    img_thres = cv2.adaptiveThreshold(img_blur, 255,1, 1, 7, 2)
    return img_thres

def find_contours(img):
    img_threshold = preprocess(img)
    contours, heirarchy = cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (255, 0, 0), 2)
    return contours

def biggest_contour(contours):
    biggest = np.array([])
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02*peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area

    return biggest, max_area

def reorder(points):
    if len(points) != 0:
        points = points.reshape((4,2))
        new_points = np.zeros((4,1,2), dtype=np.int32)
        add = points.sum(1)
        dif = np.diff(points, axis=1)
        new_points[0] = points[np.argmin(add)]
        new_points[3] = points[np.argmax(add)]
        new_points[1] = points[np.argmin(dif)]
        new_points[2] = points[np.argmax(dif)]

        return new_points

def split_Boxes(image):
    rows = np.vsplit(image, 9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for col in cols:
            #col = col[10:col.shape[0]-10, 10:col.shape[1]-5]
            boxes.append(col)

    return boxes

def get_prediction(boxes, model):
    result = []
    for box in boxes:
        img = np.asarray(box)
        img = img[4:img.shape[0]-4,4:img.shape[1]-4]
        img = cv2.resize(img, (28,28))
        img = img/255
        img = img.reshape(1,28,28,1)

        predictions = model.predict(img)
        class_index = np.argmax(predictions, axis=-1)
        prob_value = np.amax(predictions)
        #print(class_index, prob_value)
        if prob_value>0.8:
            result.append(class_index[0])
        if prob_value<0.8 and prob_value>0.6:
            np.delete(predictions,class_index[0])
            class_index = np.argmax(predictions, axis=-1)
            result.append(class_index[0])
        if prob_value<0.6:
            result.append(0)

    return result

def display_nums(img, numbers, color):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for x in range(0,9):
        for y in range(0,9):
            if numbers[(y*9)+x] != 0:
                cv2.putText(img, str(numbers[(y*9)+x]),(x*secW+int(secW/2)-10,int((y+0.8)*secH)), cv2.FONT_HERSHEY_COMPLEX, 1,color,1,cv2.LINE_AA)

    return img

# FUNCTION END

img_contour = img.copy()
img_BIG_contour = img.copy()
contours = find_contours(img_contour)
biggest, max_area = biggest_contour(contours)

if len(biggest) != 0:
    biggest = reorder(biggest)
    cv2.drawContours(img_BIG_contour,biggest,-1,(0,0,255),20)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0],[width,0],[0,height], [width,height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_warped = cv2.warpPerspective(img, matrix, (width,height))
    img_warped = cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY)


    boxes = split_Boxes(img_warped)
    numbers = get_prediction(boxes, model)
    numbers = np.asarray(numbers)

    detected_digits = blank_img.copy()
    detected_digits = display_nums(detected_digits, numbers, (255, 0, 0))

    pos_array = np.where(numbers>0,0,1)
    board = np.array_split(numbers,9)

    try:
        sudoku.solve(board)
    except:
        pass

    flatlist = []
    for rows in board:
        for r in rows:
            flatlist.append(r)

    solutions = blank_img.copy()
    solved = flatlist * pos_array
    solutions = display_nums(solutions, solved,(255,255,0))

    pts2 = np.float32(biggest)
    pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_inverse_warped = img.copy()
    img_inverse_warped = cv2.warpPerspective(solutions, matrix, (width, height))
    img_overlay = cv2.addWeighted(img_inverse_warped,1,img,0.5,1)


    imgs = [[img, img_BIG_contour,img_warped], [detected_digits,solutions,img_overlay]]

    array = stackImages(0.5,imgs)
    cv2.imshow("Img", array)

else:
    print("No sudoku found")

cv2.waitKey(0)
