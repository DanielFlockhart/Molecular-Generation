import tkinter as tk
from tkinter import Label, StringVar
from PIL import Image, ImageTk, ImageGrab
import os
class Profile:
    def __init__(self, root, mols,folder):
        self.root = root
        self.root.title("Molecule Profiles")
        self.molecule_data = mols

        self.profile_frames = []
        self.folder = folder

        row_count = 0
        col_count = 0
        max_name_length = 200
        for molecule in self.molecule_data:
            profile_frame = tk.Frame(root, borderwidth=2, relief="solid")
            profile_frame.grid(row=row_count, column=col_count, padx=10, pady=10)

            name_label = Label(profile_frame, text=molecule["name"], font=("Helvetica", 16), wraplength=max_name_length)
            name_label.pack(pady=10)

            molecule_image = Image.open(molecule["image_path"])
            molecule_image = molecule_image.resize((200, 200))
            molecule_photo = ImageTk.PhotoImage(molecule_image)

            image_label = Label(profile_frame, image=molecule_photo)
            image_label.image = molecule_photo  # Keep a reference to the image
            image_label.pack()

            stats_text = StringVar()
            stats_text.set("\n".join([f"\u2022 {key}: {value}" for key, value in molecule["stats"].items()]))
            stats_display = Label(profile_frame, textvariable=stats_text, font=("Helvetica", 12), wraplength=max_name_length, justify='left', anchor='w', padx=10)
            stats_display.pack()

            save_button = tk.Button(profile_frame, text="Save Profile", command=lambda mol=molecule, frm=profile_frame: self.save_profile_data(mol, frm))
            save_button.pack()

            self.profile_frames.append(profile_frame)

            col_count += 1
            if col_count > 7:
                col_count = 0
                row_count += 1

        save_all_button = tk.Button(root, text="Save All", command=self.save_all_profiles)
        self.root.grid_rowconfigure(row_count + 1, weight=1)  # Ensure proper layout
        save_all_button.grid(row=row_count + 1, columnspan=8) 


    def save_profile_data(self, molecule, frame):
        folder_path = os.path.join(self.folder, molecule['name'])
        os.makedirs(folder_path, exist_ok=True)  # Create folder if it doesn't exist

        x, y, width, height = frame.winfo_rootx(), frame.winfo_rooty(), frame.winfo_width(), frame.winfo_height()
        profile_image = ImageGrab.grab(bbox=(x, y, x + width, y + height))
        profile_image.save(os.path.join(folder_path, f"{molecule['name']}_profile.png"))
        print(f"Saved {molecule['name']}'s profile image.")

        # Save stats to a text file
        stats_text = "\n".join([f"{key}: {value}" for key, value in molecule["stats"].items()])
        with open(os.path.join(folder_path, f"{molecule['name']}_stats.txt"), "w") as f:
            f.write(stats_text)
            print(f"Saved {molecule['name']}'s stats to a text file.")
            print(f"Saved {molecule['name']}'s stats to a text file.")

    def save_all_profiles(self):
        for idx, molecule in enumerate(self.molecule_data):
            frame = self.profile_frames[idx]
            self.save_profile_data(molecule, frame)