import tkinter as tk
from tkinter import ttk
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from main import train_and_return_data


def start_training():
    thread = threading.Thread(target=run_training, daemon=True)
    thread.start()


def run_training():
    epochs = int(epochs_entry.get())
    hidden_dim = int(hidden_dim_entry.get())
    learning_rate = float(lr_entry.get())
    activation = activation_var.get()
    dataset = dataset_var.get()

    nn, X_train, Y_train, losses = train_and_return_data(epochs, hidden_dim, learning_rate, activation, dataset)

    root.after(0, update_plot, nn, X_train, Y_train, losses)


def update_plot(nn, X_train, Y_train, losses):
    for widget in plot_frame.winfo_children():
        widget.destroy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    from utils import plot_decision_boundary
    plot_decision_boundary(nn, X_train, Y_train, ax=axes[0])

    axes[1].plot(range(len(losses)), losses, label="Loss")
    axes[1].set_xlabel("Епохи")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Зміна Loss за епохами")
    axes[1].legend()

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    canvas.draw()


root = tk.Tk()
root.title("Налаштування нейромережі")
root.geometry("800x500")

frame_inputs = tk.Frame(root)
frame_inputs.pack(pady=10, padx=10, anchor="w")

tk.Label(frame_inputs, text="Кількість епох:").grid(row=0, column=0, sticky="w", padx=10, pady=2)
epochs_entry = tk.Entry(frame_inputs, width=10)
epochs_entry.insert(0, "5000")
epochs_entry.grid(row=0, column=1, sticky="e", padx=10, pady=2)

tk.Label(frame_inputs, text="Кількість нейронів у прихованому шарі:").grid(row=1, column=0, sticky="w", padx=10, pady=2)
hidden_dim_entry = tk.Entry(frame_inputs, width=10)
hidden_dim_entry.insert(0, "300")
hidden_dim_entry.grid(row=1, column=1, sticky="e", padx=10, pady=2)

tk.Label(frame_inputs, text="Learning Rate:").grid(row=2, column=0, sticky="w", padx=10, pady=2)
lr_entry = tk.Entry(frame_inputs, width=10)
lr_entry.insert(0, "0.2")
lr_entry.grid(row=2, column=1, sticky="e", padx=10, pady=2)

tk.Label(frame_inputs, text="Активаційна функція:").grid(row=3, column=0, sticky="w", padx=10, pady=2)
activation_var = tk.StringVar(value="relu")
activation_menu = ttk.Combobox(frame_inputs, textvariable=activation_var, values=["relu", "sigmoid", "tanh", "swish"],
                               width=8)
activation_menu.grid(row=3, column=1, sticky="e", padx=10, pady=2)

tk.Label(frame_inputs, text="Вибір датасету:").grid(row=4, column=0, sticky="w", padx=10, pady=2)
dataset_var = tk.StringVar(value="Спіраль")
dataset_menu = ttk.Combobox(frame_inputs, textvariable=dataset_var, values=["XOR", "Спіраль", "Серце"], width=8)
dataset_menu.grid(row=4, column=1, sticky="e", padx=10, pady=2)

button_frame = tk.Frame(frame_inputs)
button_frame.grid(row=5, column=1, sticky="e", padx=10, pady=10)

start_button = tk.Button(button_frame, text="Почати", command=start_training)
start_button.pack(side=tk.LEFT, padx=5)

exit_button = tk.Button(button_frame, text="Завершити роботу", command=root.quit)
exit_button.pack(side=tk.RIGHT, padx=5)

plot_frame = tk.Frame(root)
plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

root.mainloop()
