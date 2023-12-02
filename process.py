import cv2
import card
import hand

def process_image(path):
    image = cv2.imread(path)

    scale_percent = 50  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # Pre-process camera image (gray, blur, and threshold it)
    pre_proc = card.preprocess_image(image)

    # Find and sort the contours of all cards in the image (query cards)
    cnts_sort, cnt_is_card = card.find_cards(pre_proc)

    determine_hand_cards = []

    if len(cnts_sort) != 0:

        # Initialize a new "cards" list to assign the card objects.
        # k indexes the newly made array of cards.
        cards = []
        k = 0

        # For each contour detected:
        for i in range(len(cnts_sort)):
            if cnt_is_card[i] == 1:
                # Create a card object from the contour and append it to the list of cards.
                # preprocess_card function takes the card contour and contour and
                # determines the cards properties (corner points, etc.). It generates a
                # flattened 200x300 image of the card, and isolates the card's
                # suit and rank from the image.
                cards.append(card.preprocess_card(cnts_sort[i], image))

                determine_hand_cards.append([cards[k].best_rank_match, cards[k].best_suit_match])

                # Draw center point and match result on the image.
                image = card.draw_results(image, cards[k])
                k = k + 1

        # Draw card contours on image (have to do contours all at once, or
        # they do not show up properly for some reason)
        if len(cards) != 0:
            temp_cnts = []
            for i in range(len(cards)):
                temp_cnts.append(cards[i].contour)
            cv2.drawContours(image, temp_cnts, -1, (255, 0, 0), 2)

    print(determine_hand_cards)
    return image, hand.determine_poker_hand(determine_hand_cards)
    # print(hand.determine_poker_hand(determine_hand_cards))
    # Finally, display the image with the identified cards!
    # cv2.imshow("Card Detector", image)
    # cv2.waitKey(0)
    #
    # # Close all windows and close the PiCamera video stream.
    # cv2.destroyAllWindows()
