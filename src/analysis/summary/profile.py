import tkinter as tk
from tkinter import Label, StringVar
from PIL import Image, ImageTk

class Profile:
    def __init__(self, root, mols):
        self.root = root
        self.root.title("Molecule Profiles")
        self.molecule_data = mols

        self.profile_frames = []

        for molecule in self.molecule_data:
            profile_frame = tk.Frame(root, borderwidth=2, relief="solid")
            profile_frame.pack(side="left", padx=10, pady=10)

            name_label = Label(profile_frame, text=molecule["name"], font=("Helvetica", 16))
            name_label.pack(pady=10)

            molecule_image = Image.open(molecule["image_path"])
            molecule_image = molecule_image.resize((100, 100))
            molecule_photo = ImageTk.PhotoImage(molecule_image)

            image_label = Label(profile_frame, image=molecule_photo)
            image_label.image = molecule_photo  # Keep a reference to the image
            image_label.pack()

            stats_text = StringVar()
            stats_text.set("\n".join([f"{key}: {value}" for key, value in molecule["stats"].items()]))
            stats_display = Label(profile_frame, textvariable=stats_text, font=("Helvetica", 12))
            stats_display.pack()

            self.profile_frames.append(profile_frame)