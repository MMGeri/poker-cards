import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

# signs / values
dataToSegment = 'signs'
trainOrTest = "train"

input_folder = 'C:\\Users\\mager\\Desktop\\poker-cards2\\ML\\augment_data\\inputs'
imagePath = f"{input_folder}\\{trainOrTest}_zipped\\"
output_folder = 'C:\\Users\\mager\\Desktop\\poker-cards2\\ML\\training\\data'

# filename,width,height,class,xmin,ymin,xmax,ymax
train_df = pd.read_csv(f"{input_folder}\\{trainOrTest}_cards_label.csv", dtype={'label': str})
train_df.head()

i = 0
rowsToProcess = len(train_df)
image = cv2.imread(f"{imagePath}\\{train_df['filename'][i]}", cv2.IMREAD_GRAYSCALE)

def getRectanglesOnImage(i):
    cFilename = train_df['filename'][i]
    rectanglesOnImage = []
    labels = []
    filenames = []

    while cFilename == train_df['filename'][i]:
        xmin = train_df['xmin'][i]
        ymin = train_df['ymin'][i]
        xmax = train_df['xmax'][i]
        ymax = train_df['ymax'][i]

        rectanglesOnImage.append([xmin, ymin, xmax, ymax])
        labels.append(train_df['class'][i])
        filenames.append(train_df['filename'][i])
        i += 1

        if i >= rowsToProcess:
            break
    return [rectanglesOnImage, i, labels, filenames]

def getCentroid(rectangles):
    centreOfRectangles = []
    for rectangle in rectangles:
        centreOfRectangles.append([(rectangle[0] + rectangle[2]) / 2, (rectangle[1] + rectangle[3]) / 2])

    # centroid = np.mean(centreOfRectangles, axis=0)
    centroid = [400,750]
    return (centroid,centreOfRectangles)

def getAngleOfBox(centreOfRectangle, centroid):
    im_height, im_width = image.shape

    centroid[1] = im_height - centroid[1]
    centreOfRectangle[1] = im_height - centreOfRectangle[1]

    vector_1 = np.array(centreOfRectangle) - np.array(centroid)
    vector_2 = np.array([0, 1])

    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    angle = angle + (angle / np.pi / 10)
    if centreOfRectangle[0] > centroid[0]:
        angle = -angle

    return angle

def cutOutBoxFromBottom(rotatedRectangle, dataToSegment):
    if(dataToSegment == 'signs'):
        IMAGE = rotatedRectangle[rotatedRectangle.shape[0] - 34:rotatedRectangle.shape[0] - 3,
                 int(rotatedRectangle.shape[1] / 2) - 12:int(rotatedRectangle.shape[1] / 2) + 15]
    else:
        IMAGE = rotatedRectangle[0:rotatedRectangle.shape[0] - 24,
                int(rotatedRectangle.shape[1] / 2) - 12:int(rotatedRectangle.shape[1] / 2) + 15]
    return IMAGE

def rotate_image(mat, angle):
    angle = angle * 180 / np.pi

    height, width = mat.shape[:2]  # image shape has 3 dimensions
    image_center = (
    width / 2, height / 2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat

def filterOutDuplicates(rectanglesOnImage):
    # if a rectangles label is twice then only use the first rectangle
    filteredLabels = []
    filteredRectangles = []
    for rectangle in rectanglesOnImage:
        if rectangle['label'] not in filteredLabels:
            filteredLabels.append(rectangle['label'])
            filteredRectangles.append(rectangle)
    return filteredRectangles

def fillClearBlackAreasWithWhite(SIGN):
    mask = cv2.inRange(SIGN, 0, 5)
    med = np.nanmedian(SIGN)

    # convert mask to 0-1 values
    mask[mask != 0] = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilate = cv2.dilate(mask, kernel, iterations=1)

    SIGN[dilate != 0] = med
    return SIGN

#make directory if it doesn't exist
if not os.path.exists(f"{output_folder}\\{dataToSegment}"):
    os.makedirs(f"{output_folder}\\{dataToSegment}")

# create or override the csv file
with open(f"{output_folder}\\{dataToSegment}.csv", "w") as file:
    file.write("filename,label\n")

while i < rowsToProcess:
    image = cv2.imread(f"{imagePath}\\{train_df['filename'][i]}", cv2.IMREAD_GRAYSCALE)
    # get rectangles on image
    rectanglesOnImage, i, labels, filenames = getRectanglesOnImage(i)
    # get centroid
    cc, centreOfRectangles = getCentroid(rectanglesOnImage)
    # give each rectangle the centre property
    mapped_objects = []
    for rectangle, centre, label, filename in zip(rectanglesOnImage, centreOfRectangles, labels, filenames):
        # Create an object with coordinates and centre properties
        rectangle_object = {
            "filename": filename,
            "coordinates": rectangle,
            "centre": centre,
            "label": label
        }

        # Append the object to the list
        mapped_objects.append(rectangle_object)
    rectanglesOnImage = mapped_objects
    rectanglesOnImage = filterOutDuplicates(rectanglesOnImage)
    for rectangle in rectanglesOnImage:
        # the angle from the centroid to the centre of the rectangle
        angle = getAngleOfBox(rectangle['centre'], cc.copy())
        # rotate the rectangle
        coords = rectangle['coordinates']
        rotatedRectangle = rotate_image(image[coords[1]:coords[3], coords[0]:coords[2]], -angle)

        IMG = cutOutBoxFromBottom(rotatedRectangle, dataToSegment)
        IMG = fillClearBlackAreasWithWhite(IMG)

        # save the image and the image-label csv to the output folder
        # make it 28 by 28 and stretch it if it needs to be stretched
        IMG = cv2.resize(IMG, (28, 28), interpolation=cv2.INTER_AREA)

        cv2.imwrite(
            f"{output_folder}\\{dataToSegment}_{trainOrTest}\\{rectangle['filename'].rstrip('.jpg')}_{rectangle['label']}.jpg",
            IMG)

        if(dataToSegment == 'signs'):
            with open(f"{output_folder}\\{dataToSegment}_{trainOrTest}.csv", "a") as file:
                file.write(f"{rectangle['filename'].rstrip('.jpg')}_{rectangle['label']}.jpg,{rectangle['label'][-1]}\n")
        else:
            with open(f"{output_folder}\\{dataToSegment}_{trainOrTest}.csv", "a") as file:
                file.write(f"{rectangle['filename'].rstrip('.jpg')}_{rectangle['label']}.jpg,{rectangle['label'][:-1]}\n")


#show image
