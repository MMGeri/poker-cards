import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import card
import process
from segment import findAllContours

class PokerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Poker Hand Detector")
        self.root.geometry("1000x620")

        # Create a frame for widgets
        self.widget_frame = tk.Frame(self.root)
        self.widget_frame.pack(side=tk.LEFT, fill=tk.Y)

        # Create a frame for the image
        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Variables for storing constants
        self.bkg_thresh_var = tk.StringVar()
        self.card_thresh_var = tk.StringVar()
        self.card_max_area = tk.StringVar()
        self.card_min_area = tk.StringVar()
        self.rank_height_af_min = tk.IntVar()
        self.rank_height_af_max = tk.IntVar()

        # Set initial values for constants
        self.bkg_thresh_var.set(str(card.BKG_THRESH))
        self.card_thresh_var.set(str(card.CARD_THRESH))
        self.card_max_area.set(str(card.CARD_MAX_AREA))
        self.card_min_area.set(str(card.CARD_MIN_AREA))
        self.rank_height_af_min.set(int(10))
        self.rank_height_af_max.set(int(50))

        # File path of the selected image
        self.image_path = None

        # Create UI elements
        self.create_widgets()
        self.create_image_display()

    def create_widgets(self):
        # File selection button
        self.browse_button = tk.Button(self.widget_frame, text="Browse Image", command=self.browse_image)
        self.browse_button.pack()

        # Constants modification
        self.create_entry("Background threshold level:", self.bkg_thresh_var)
        self.create_entry("Card threshold level:", self.card_thresh_var)
        self.create_entry("Card minimal area:", self.card_min_area)
        self.create_entry("Card maximal area:", self.card_max_area)


        # Process and display button
        self.process_button = tk.Button(self.widget_frame, text="Process Image", command=self.process_image)
        self.process_button.pack()

        # Result display
        self.result_label = tk.Label(self.widget_frame, text="")
        self.result_label.pack()

        # Additional button for the new function
        self.create_entry("Rank height min for af:", self.rank_height_af_min)
        self.create_entry("Rank height max for af:", self.rank_height_af_max)
        self.run_additional_function_button = tk.Button(self.widget_frame, text="Run Additional Function", command=self.run_additional_function)
        self.run_additional_function_button.pack()



    def create_entry(self, label_text, variable):
        label = tk.Label(self.widget_frame, text=label_text)
        label.pack()
        entry = tk.Entry(self.widget_frame, textvariable=variable)
        entry.pack()

    def create_image_display(self):
        # Image display
        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack()

    def browse_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.image_path = file_path
            self.show_image()

    def show_image(self):
        if self.image_path:
            # Load and resize the image
            image = cv2.imread(self.image_path)
            image = cv2.resize(image, (800, 600))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(image)
            img_tk = ImageTk.PhotoImage(img)

            # Update the image label
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk

    def process_image(self):
        if self.image_path:
            # Update constants with user-modified values
            card.BKG_THRESH = int(self.bkg_thresh_var.get())
            card.CARD_THRESH = int(self.card_thresh_var.get())
            card.CARD_MAX_AREA = int(self.card_max_area.get())
            card.CARD_MIN_AREA = int(self.card_min_area.get())

            # Process the image using process.py
            result_image, poker_hand_result = process.process_image(self.image_path)

            # Display the processed image
            self.show_processed_image(result_image)

            self.result_label.config(text=f"Poker Hand: {poker_hand_result}")

    def show_processed_image(self, result_image):
        # Resize the processed image
        result_image = cv2.resize(result_image, (800, 600))
        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(result_image)
        img_tk = ImageTk.PhotoImage(img)

        # Display the processed image
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk

    def run_additional_function(self):
        # This is a placeholder function, replace it with your actual logic
        additional_result = findAllContours(cv2.imread(self.image_path), self.rank_height_af_min.get(), self.rank_height_af_max.get(), self.bkg_thresh_var.get())
        self.show_processed_image(additional_result)  # Example: Show an additional image


if __name__ == "__main__":
    root = tk.Tk()
    app = PokerApp(root)
    root.mainloop()
