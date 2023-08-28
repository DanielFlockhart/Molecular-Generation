import sys,os
from profile import Profile
import tkinter as tk
sys.path.insert(0, os.path.abspath('../..'))
from Constants import file_constants

counter = 16
molecule_data = []
for file in os.listdir(r"C:\Users\0xdan\Documents\CS\WorkCareer\Chemistry Internship\Project-Code\data\datasets\db2-pas\Targets"):
    if file.endswith(".png"):
        counter-=1
        molecule_data.append({
            "name": file[:-4],
            "image_path": os.path.join(r"C:\Users\0xdan\Documents\CS\WorkCareer\Chemistry Internship\Project-Code\data\datasets\db2-pas\Targets", file),
            "stats": {
                "Molecular Formula": "CH4",
                "Molar Mass": "16.04 g/mol",
                "Melting Point": "-182.5°C",
                "Boiling Point": "-161.5°C",
            }
        })
        if counter == 0:
            break

if __name__ == "__main__":
    root = tk.Tk()
    app = Profile(root,molecule_data,file_constants.PROFILES_FOLDER)
    root.mainloop()

