from pytesseract import pytesseract
import os
import cv2
import numpy as np

def list_files(directory, extensions=None, shuffle=False):
    """
    Lists files in a directory
    :return:
    """

    images = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(subdir, file)

            if extensions is not None:
                if file_path.endswith(tuple(extensions)):
                    images.append(file_path)
            else:
                images.append(file_path)
    if shuffle:
        np.random.shuffle(images)
    return images


def edge_detection(image):
    blur = cv2.GaussianBlur(image, (3, 3), 0)
    edged = cv2.Canny(blur, 0, 255)
    edged = cv2.dilate(edged, (3, 3), iterations=2)
    edged = cv2.erode(edged, (3, 3), iterations=1)
    return edged

def find_bbox(edged_image):
    contours, _ = cv2.findContours(edged_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(image, contours, -1, (0, 255, 255), 1)
    bboxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bboxes.append([x,y,x+w,y+h])
    return bboxes

def bbox_filter(image,bboxes):
    image_ratio = image.shape[0] / image.shape[1]
    filter_boxes = []
    for box in bboxes:
        x,y,w,h = box
        cropped_box = image[y:h,x:w]
        if image_ratio < 0.53 and image_ratio > 0.48:
            if ((cropped_box.shape[0] / image.shape[0]) > 0.23) and (cropped_box.shape[1] > 25):
                filter_boxes.append([x,y,w,h])

        elif image_ratio < 0.72 and image_ratio > 0.68:
            if ((cropped_box.shape[0] / image.shape[0]) > 0.22) and (cropped_box.shape[1] > 34):
                filter_boxes.append([x,y,w,h])


        elif image_ratio < 0.3 and image_ratio > 0.23:
            if ((cropped_box.shape[0] / image.shape[0]) > 0.6) and (cropped_box.shape[1] > 17):
                filter_boxes.append([x,y,w,h])

        elif image_ratio < 0.43 and image_ratio > 0.32:
            if ((cropped_box.shape[0] / image.shape[0]) > 0.2) and (cropped_box.shape[1] > 13):
                filter_boxes.append([x, y, w, h])

    # print("Filter :",filter_boxes)
    return filter_boxes

if __name__ == '__main__':
    # provide path of image
    data_dir = "/home/pavesh/Desktop/aligned_images"
    images = list_files(directory=data_dir, shuffle= True, extensions=[".jpg", ".png"])
    for img in images:
        image = cv2.imread(img)
        edged = edge_detection(image)
        bboxes = find_bbox(edged)
        filter_boxes = bbox_filter(image,bboxes)
        for box in filter_boxes:
            # segmented box coordinates
            x,y,w,h = box
            cv2.rectangle(image,(x,y),(w,h),(160,240,255),1)

            # cropping character using box coordinates
            box_crop = image[y:h,x:w]
            box_gray = cv2.cvtColor(box_crop,cv2.COLOR_BGR2GRAY)
            box_thresh = cv2.adaptiveThreshold(box_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,21,4)
            text = pytesseract.image_to_string(box_crop,lang="lnp",config="-- psm 10")
            print(text)
        # cv2.imshow("thresh",box)
        # cv2.imshow("Segment",image)
        # cv2.waitKey(0)



