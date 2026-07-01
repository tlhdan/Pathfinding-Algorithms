import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tkinter as tk
from tkinter import ttk
from maze_generator import create_maze, generate_bonuses, generate_obstacles
import visualization.static_viz as static_viz
import visualization.dynamic_viz as dynamic_viz

def select(event):
    selected = combo_box.get()
    if selected == "Static maze":
        static_panel()
    else:
        dynamic_panel()

def run():
    selected = combo_box.get()
    rows = rows_scale.get()
    cols = cols_scale.get()
    density = density_scale.get()

    if selected == "Static maze" and weighted_var.get() == 1:
        weighted = True
    
    else:
        weighted = False

    maze, start, finish = create_maze(rows, cols, density, weighted, min_sf_distance=(rows + cols) // 4)

    if selected == "Static maze":
        static_viz.visualize(maze, start, finish)

    else:
        num_bonuses = bonuses_scale.get()
        bonuses = generate_bonuses(maze, start, finish, num_bonuses)

        num_obstacles = obstacles_scale.get()
        obstacles = generate_obstacles(maze, start, finish, num_obstacles)

        dynamic_viz.visualize(maze, start, finish, bonuses, obstacles)

def static_panel():
    rows_scale.config(from_=3, to=99)
    rows_scale.grid(row=3, column=1)

    cols_scale.config(from_=3, to=99)
    cols_scale.grid(row=4, column=1)

    weighted_checkbox.grid(row=6, column=0, columnspan=2)

    bonuses_label.grid_forget()
    bonuses_scale.grid_forget()
    
    obstacles_label.grid_forget()
    obstacles_scale.grid_forget()

    run_button.grid(row=7, column=0)

    exit_button.grid(row=7, column=1)

def dynamic_panel():
    rows_scale.config(from_=5, to=15)

    cols_scale.config(from_=5, to=15)

    weighted_checkbox.grid_forget()

    bonuses_label.grid(row=6, column=0)
    bonuses_scale.grid(row=6, column=1)

    obstacles_label.grid(row=7, column=0)
    obstacles_scale.grid(row=7, column=1)

    run_button.grid(row=8, column=0)

    exit_button.grid(row=8, column=1)
    
root = tk.Tk()
root.title("Pathfinding Algorithm Visualization")

tk.Label(root, text="Select visualization:").grid(row=0, column=0, columnspan=2)

combo_box = ttk.Combobox(
    root,
    values=["Static maze", "Dynamic maze"],
    state="readonly"
)
combo_box.grid(row=1, column=0, columnspan=2)
combo_box.set("Static maze")
combo_box.bind("<<ComboboxSelected>>", select)

tk.Label(root, text="Rows").grid(row=3, column=0)
rows_scale = tk.Scale(root, from_=3, to=99, resolution=2, orient="horizontal")

tk.Label(root, text="Columns").grid(row=4, column=0)
cols_scale = tk.Scale(root, from_=3, to=99, resolution=2, orient="horizontal")

tk.Label(root, text="Wall density").grid(row=5, column=0)
density_scale = tk.Scale(root, from_=0, to=1, resolution=0.01, orient="horizontal")
density_scale.set(1)
density_scale.grid(row=5, column=1)

weighted_var = tk.IntVar()
weighted_checkbox = tk.Checkbutton(root, text='Weighted maze',variable=weighted_var, onvalue=1, offvalue=0)
weighted_checkbox.grid(row=6, column=0, columnspan=2)

bonuses_label = tk.Label(root, text="Bonuses")
bonuses_scale = tk.Scale(root, from_=0, to=5, orient="horizontal")

obstacles_label = tk.Label(root, text="Obstacles")
obstacles_scale = tk.Scale(root, from_=0, to=5, orient="horizontal")

run_button = tk.Button(root, text="Run", bg="lightgreen", activebackground="lightgreen", command=run)

exit_button = tk.Button(root, text="Exit", foreground="white", activeforeground="white", bg="red", activebackground="red", command=root.destroy)

static_panel()

root.mainloop()