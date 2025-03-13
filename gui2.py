from tkinter import *
from PIL import Image as PIL_Image
from PIL import ImageTk
import tkinter.filedialog as tkFileDialog
import cv2
from tkinter import messagebox
import matplotlib.pyplot as plt


# Retrieving the data stored by the previously running notebooks 

# Result list: contains complete result analysis of each feaature
%store -r result_list         

# Path of input image
%store -r path     


# Function to display the output
def display_output():
    # Creating sub-frames inside the master_frame
    sub_frames = [Frame(master_frame, bg=color, pady=5, padx=5) for color in ['black', 'brown', None, None]]

    # Placing sub-frames in a grid layout
    for i, frame in enumerate(sub_frames, start=1):
        frame.grid(row=i, column=1, padx=5, pady=5)

    # Title label in sub_frame1
    Label(master=sub_frames[0], text="FAKE CURRENCY DETECTION SYSTEM", fg='dark blue', font="Verdana 28 bold").pack()

    # Canvas for input image in sub_frame2
    canvas_input = Canvas(master=sub_frames[1], width=675, height=300)
    canvas_input.pack()

    if len(path) > 0 and path.endswith('.jpg'):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (675, 300))
        image = ImageTk.PhotoImage(PIL_Image.fromarray(image))

        canvas_input.image = image
        canvas_input.create_image(0, 0, anchor=NW, image=image)

    pass_count = 0

    # Display feature analysis in sub_frame4
    for i in range(4):
        for j in range(3):
            feature_num = 3 * i + j
            if feature_num >= 10:
                break

            # Creating a feature frame
            feature_frame = Frame(sub_frames[3], relief=RAISED, borderwidth=1, bg='light blue')
            feature_frame.grid(row=i, column=j, padx=20, pady=20, sticky="nsew")

            # Inner frames for details
            frames = [Frame(feature_frame, padx=3, pady=3),
                      Frame(feature_frame, bg='brown', pady=5, padx=5),
                      Frame(feature_frame),
                      Frame(feature_frame),
                      Frame(feature_frame)]

            for idx, frame in enumerate(frames, start=1):
                frame.grid(row=idx, column=1, padx=5, pady=5)

            # Feature label
            Label(frames[0], text=f"Feature {feature_num + 1}", fg='black', font="Verdana 12 bold").pack()

            # Display feature image
            canvas = Canvas(frames[1], width=200, height=200)
            canvas.pack()
            image = result_list[feature_num][0].copy()

            # Maintain aspect ratio
            h, w = image.shape[:2]
            aspect_ratio = w / h
            resize_width, resize_height = (200, int(200 / aspect_ratio)) if w > h else (int(200 * aspect_ratio), 200)
            image = cv2.resize(image, (resize_width, resize_height))
            image = ImageTk.PhotoImage(PIL_Image.fromarray(image))

            canvas.image = image
            canvas.create_image((200-resize_width)//2, (200-resize_height)//2, anchor=NW, image=image)

            # Analysis details
            if feature_num < 7:
                text2 = f"Avg. SSIM Score: {result_list[feature_num][1]:.3f}"
                text3 = f"Max. SSIM Score: {result_list[feature_num][2]:.3f}"
            elif feature_num < 9:
                text2 = f"Avg. Number of lines: {result_list[feature_num][1]:.3f}"
                text3 = ""
            else:
                text2 = "9 characters detected!" if result_list[feature_num][1] else "Less than 9 characters detected!"
                text3 = ""

            Label(frames[2], text=text2, fg='dark blue', font="Verdana 11", bg='light blue').pack()
            Label(frames[3], text=text3, fg='dark blue', font="Verdana 11", bg='light blue').pack()

            # Status Pass/Fail
            status = result_list[feature_num][3 if feature_num < 7 else 2]
            pass_count += 1 if status else 0
            status_text = "PASS!" if status else "FAIL!"
            status_color = 'green' if status else 'red'

            Label(frames[4], text=f"Status: {status_text}", fg=status_color, font="Verdana 11 bold", bg='light blue').pack()

    # Final result in sub_frame3
    Label(master=sub_frames[2], text=f"RESULT: {pass_count} / 10 features PASSED!", fg='green', font="Verdana 24 bold").pack()


# Scrollbar functionality
def scrollbar_function(event):
    canvas.configure(scrollregion=canvas.bbox("all"), width=1050, height=550)


# Main application window
root = Tk()
root.title('Fake Currency Detection - Result Analysis')
root.resizable(False, False)

window_width, window_height = 1100, 600
x, y = (root.winfo_screenwidth() - window_width) // 2, (root.winfo_screenheight() - window_height) // 2
root.geometry(f"{window_width}x{window_height}+{x}+{y}")

# Main frame and canvas
main_frame = Frame(root, relief=GROOVE, bd=1)
main_frame.place(x=10, y=10)

canvas = Canvas(main_frame)
master_frame = Frame(canvas)

# Scrollbar
myscrollbar = Scrollbar(main_frame, orient="vertical", command=canvas.yview)
canvas.configure(yscrollcommand=myscrollbar.set)
myscrollbar.pack(side="right", fill="y")
canvas.pack(side="left")

canvas.create_window((0, 0), window=master_frame, anchor='nw')
master_frame.bind("<Configure>", scrollbar_function)

# Display output
display_output()

# Run the application
root.mainloop()