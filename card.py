import tensorflow as tf
import numpy as np
import cv2

# Constants

# Adaptive threshold levels
BKG_THRESH = 10
CARD_THRESH = 30

# Width and height of card corner, where rank and suit are
CORNER_WIDTH = 32
CORNER_HEIGHT = 84

# Dimensions of rank train images
RANK_WIDTH = 70
RANK_HEIGHT = 125

# Dimensions of suit train images
SUIT_WIDTH = 70
SUIT_HEIGHT = 100

CARD_MAX_AREA = 600000
CARD_MIN_AREA = 25000

font = cv2.FONT_HERSHEY_SIMPLEX

numbers_model = tf.keras.models.load_model("ML/models/backup/number_model_2.0.h5")
rank_model = tf.keras.models.load_model("ML/models/backup/values_model_3.0.h5")
suit_model = tf.keras.models.load_model("ML/models/backup/signs_model_2.0.h5")

def predict_rank(image):
    image = cv2.resize(image, (28, 28))

    # dilate image with opencv
    image = cv2.bitwise_not(image)

    test_images = np.array(image)
    test_images = np.array([test_images])  # Convert single image to a batch.
    test_images = test_images.astype('float32')
    test_images /= 255

    predictions_v = rank_model.predict(test_images)
    predictions_n = numbers_model.predict(test_images)
    classes = ['A', 'J', 'K', 'Q', '']
    print("betu ", np.max(predictions_v[0]), "szam",  np.max(predictions_n[0]))
    if np.max(predictions_v[0]) > 0.9 and classes[np.argmax(predictions_v[0])] != '':
        return classes[np.argmax(predictions_v[0])]
    return np.argmax(predictions_n[0])


def predict_suit(image):
    model = suit_model
    image = cv2.resize(image, (28, 28))

    # dilate image with opencv
    image = cv2.bitwise_not(image)

    test_images = np.array(image)
    test_images = np.array([test_images])  # Convert single image to a batch.
    test_images = test_images.astype('float32')
    test_images /= 255

    predictions = model.predict(test_images)
    classes = ['c', 'd', 'h', 's']
    return classes[np.argmax(predictions[0])]

# Structures to hold query card and train card information
class Card:
    def __init__(self):
        self.contour = []  # Contour of card
        self.width, self.height = 0, 0  # Width and height of card
        self.corner_pts = []  # Corner points of card
        self.center = []  # Center point of card
        self.warp = []  # 200x300, flattened, grayed, blurred image
        self.rank_img = []  # Thresholded, sized image of card's rank
        self.suit_img = []  # Thresholded, sized image of card's suit
        self.best_rank_match = "Unknown"  # Best matched rank
        self.best_suit_match = "Unknown"  # Best matched suit


def preprocess_image(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    img_w, img_h = np.shape(image)[:2]
    bkg_level = gray[int(img_h / 100)][int(img_w / 2)]
    thresh_level = bkg_level + BKG_THRESH

    retval, thresh = cv2.threshold(blur, thresh_level, 255, cv2.THRESH_BINARY)
    return thresh


def find_cards(thresh_image):

    # Find contours and sort their indices by contour size
    contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    index_sort = sorted(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]), reverse=True)

    # If there are no contours, do nothing
    if len(contours) == 0:
        return [], []

    # Otherwise, initialize empty sorted contour and hierarchy lists
    contours_sorted = []
    hierarchy_sorted = []
    cnt_is_card = np.zeros(len(contours), dtype=int)

    # Fill empty lists with sorted contour and sorted hierarchy. Now,
    # the indices of the contour list still correspond with those of
    # the hierarchy list. The hierarchy array can be used to check if
    # the contours have parents or not.
    for i in index_sort:
        contours_sorted.append(contours[i])
        hierarchy_sorted.append(hierarchy[0][i])

    # Determine which of the contours are cards by applying the
    # following criteria: 1) Smaller area than the maximum card size,
    # 2), bigger area than the minimum card size, 3) have no parents,
    # and 4) have four corners

    for i in range(len(contours_sorted)):
        size = cv2.contourArea(contours_sorted[i])
        peri = cv2.arcLength(contours_sorted[i], True)
        approx = cv2.approxPolyDP(contours_sorted[i], 0.01 * peri, True)

        if ((size < CARD_MAX_AREA) and (size > CARD_MIN_AREA)
                and (hierarchy_sorted[i][3] == -1) and (len(approx) == 4)):
            cnt_is_card[i] = 1

    return contours_sorted, cnt_is_card


def preprocess_card(contour, image):

    # Initialize new Card object
    card = Card()

    card.contour = contour

    # Find perimeter of card and use it to approximate corner points
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.01 * peri, True)
    pts = np.float32(approx)
    card.corner_pts = pts

    # Find width and height of card's bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    card.width, card.height = w, h

    average = np.sum(pts, axis=0) / len(pts)
    cent_x = int(average[0][0])
    cent_y = int(average[0][1])
    card.center = [cent_x, cent_y]

    # Warp card into 200x300 flattened image using perspective transform
    card.warp = flattener(image, pts, w, h)

    # Grab corner of warped card image and do a 4x zoom
    card_corner = card.warp[0:CORNER_HEIGHT, 0:CORNER_WIDTH]
    card_corner_zoom = cv2.resize(card_corner, (0, 0), fx=4, fy=4)

    # Sample known white pixel intensity to determine good threshold level
    white_level = card_corner_zoom[15, int((CORNER_WIDTH * 4) / 2)]
    thresh_level = white_level - CARD_THRESH
    if thresh_level <= 0:
        thresh_level = 1
    retval, query_thresh = cv2.threshold(card_corner_zoom, thresh_level, 255, cv2.THRESH_BINARY_INV)

    # Split into top and bottom half (top shows rank, bottom shows suit)
    card_rank = query_thresh[20:185, 0:128]
    card_suit = query_thresh[186:336, 0:128]

    # Find rank contour and bounding rectangle, isolate and find the largest contour
    card_rank_contours, hier = cv2.findContours(card_rank, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    card_rank_contours = sorted(card_rank_contours, key=cv2.contourArea, reverse=True)

    # Find bounding rectangle for largest contour, use it to resize query rank
    # image to match dimensions of the train rank image
    if len(card_rank_contours) != 0:
        x1, y1, w1, h1 = cv2.boundingRect(card_rank_contours[0])
        card_rank_roi = card_rank[y1:y1 + h1, x1:x1 + w1]
        card_rank_sized = cv2.resize(card_rank_roi, (RANK_WIDTH, RANK_HEIGHT), 0, 0)
        card.rank_img = card_rank_sized

    # Find suit contour and bounding rectangle, isolate and find the largest contour
    card_suit_contours, hier = cv2.findContours(card_suit, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    card_suit_contours = sorted(card_suit_contours, key=cv2.contourArea, reverse=True)

    # Find bounding rectangle for largest contour, use it to resize query suit
    # image to match dimensions of the train suit image
    if len(card_suit_contours) != 0:
        x2, y2, w2, h2 = cv2.boundingRect(card_suit_contours[0])
        card_suit_roi = card_suit[y2:y2 + h2, x2:x2 + w2]
        card_suit_sized = cv2.resize(card_suit_roi, (SUIT_WIDTH, SUIT_HEIGHT), 0, 0)
        card.suit_img = card_suit_sized

    card.best_suit_match = predict_suit(card.suit_img)
    card.best_rank_match = str(predict_rank(card.rank_img))

    return card


def draw_results(image, card):
    x = card.center[0]
    y = card.center[1]

    rank_name = card.best_rank_match
    suit_name = card.best_suit_match

    # Draw card name twice, so letters have black outline
    cv2.putText(image, (rank_name + ' of'), (x - 60, y - 10), font, 1, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, (rank_name + ' of'), (x - 60, y - 10), font, 1, (50, 200, 200), 2, cv2.LINE_AA)

    cv2.putText(image, suit_name, (x - 60, y + 25), font, 1, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, suit_name, (x - 60, y + 25), font, 1, (50, 200, 200), 2, cv2.LINE_AA)

    # Can draw difference value for troubleshooting purposes
    # (commented out during normal operation)
    # r_diff = str(qCard.rank_diff)
    # s_diff = str(qCard.suit_diff)
    # cv2.putText(image,r_diff,(x+20,y+30),font,0.5,(0,0,255),1,cv2.LINE_AA)
    # cv2.putText(image,s_diff,(x+20,y+50),font,0.5,(0,0,255),1,cv2.LINE_AA)

    return image

def flattener(image, pts, w, h):
    # www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example
    temp_rect = np.zeros((4, 2), dtype="float32")

    s = np.sum(pts, axis=2)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    diff = np.diff(pts, axis=-1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    # Need to create an array listing points in order of
    # [top left, top right, bottom right, bottom left]
    # before doing the perspective transform

    if w <= 0.8 * h:  # If card is vertically oriented
        temp_rect[0] = tl
        temp_rect[1] = tr
        temp_rect[2] = br
        temp_rect[3] = bl

    if w >= 1.2 * h:  # If card is horizontally oriented
        temp_rect[0] = bl
        temp_rect[1] = tl
        temp_rect[2] = tr
        temp_rect[3] = br

    # If the card is 'diamond' oriented, a different algorithm
    # has to be used to identify which point is top left, top right
    # bottom left, and bottom right.

    if 0.8 * h < w < 1.2 * h:  # If card is diamond oriented
        # If furthest left point is higher than furthest right point,
        # card is tilted to the left.
        if pts[1][0][1] <= pts[3][0][1]:
            # If card is titled to the left, approxPolyDP returns points
            # in this order: top right, top left, bottom left, bottom right
            temp_rect[0] = pts[1][0]  # Top left
            temp_rect[1] = pts[0][0]  # Top right
            temp_rect[2] = pts[3][0]  # Bottom right
            temp_rect[3] = pts[2][0]  # Bottom left

        # If furthest left point is lower than furthest right point,
        # card is tilted to the right
        if pts[1][0][1] > pts[3][0][1]:
            # If card is titled to the right, approxPolyDP returns points
            # in this order: top left, bottom left, bottom right, top right
            temp_rect[0] = pts[0][0]  # Top left
            temp_rect[1] = pts[3][0]  # Top right
            temp_rect[2] = pts[2][0]  # Bottom right
            temp_rect[3] = pts[1][0]  # Bottom left

    max_width = 200
    max_height = 300

    # Create destination array, calculate perspective transform matrix,
    # and warp card image
    dst = np.array([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]], np.float32)
    perspective_transform = cv2.getPerspectiveTransform(temp_rect, dst)
    warp = cv2.warpPerspective(image, perspective_transform, (max_width, max_height))
    warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)

    return warp


