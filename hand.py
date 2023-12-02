from collections import Counter
import card


def face_card_value(rank):
    face_cards = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '0': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
    return face_cards.get(rank, rank)


def determine_poker_hand(cards):
    # Input example: [["K", "D"], ["J", "C"], ["A", "C"], ["10", "C"], ["Q", "C"]]
    # Sort the cards by rank

    for i in range(len(cards)):
        if cards[i][0] == "Unknown" or cards[i][1] == "Unknown":
            return "Invalid poker hand"

    cards = [[face_card_value(rank), suit] for rank, suit in cards]

    sorted_cards = sorted(cards, key=lambda x: x[0])

    # Separate ranks and suits
    ranks = [card[0] for card in sorted_cards]
    suits = [card[1] for card in sorted_cards]

    # Count occurrences of each rank
    rank_counts = Counter(ranks)

    # Check for different poker hands
    if len(cards) != 5:
        return "Invalid poker hand"
    elif all(rank in ranks for rank in [10, 11, 12, 13, 14]) and all(suit == suits[0] for suit in suits):
        return "Royal Flush"
    elif all(suit == suits[0] for suit in suits) and all(rank == ranks[0] + i for i, rank in enumerate(ranks)):
        return "Straight Flush"
    elif any(count == 4 for count in rank_counts.values()):
        return "Four of a Kind"
    elif all(count in (2, 3) for count in rank_counts.values()):
        return "Full House"
    elif all(suit == suits[0] for suit in suits):
        return "Flush"
    elif all(rank == ranks[0] + i for i, rank in enumerate(ranks)):
        return "Straight"
    elif any(count == 3 for count in rank_counts.values()):
        return "Three of a Kind"
    elif len(rank_counts) == 3 and list(rank_counts.values()).count(2) == 2:
        return "Two Pair"
    elif 2 in rank_counts.values():
        return "One Pair"
    else:
        return "High Card"

