import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class LinearRegressionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Linear Regression App")
        self.geometry("800x600")
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        self.csv_file = "10_points.csv"
        self.model_file = "our_model.pkl"

        self.sidebar_frame = ctk.CTkFrame(self, width=200)
        self.sidebar_frame.grid(row=0, column=0, sticky="ns", padx=10, pady=10)

        self.sidebar_button_train = ctk.CTkButton(
            self.sidebar_frame, text="Train", command=self.train_model
        )
        self.sidebar_button_train.pack(pady=10)

        self.sidebar_button_predict = ctk.CTkButton(
            self.sidebar_frame, text="Predict", command=self.predict_value
        )
        self.sidebar_button_predict.pack(pady=10)

        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self.label_x = ctk.CTkLabel(self.main_frame, text="Enter x:")
        self.label_x.grid(row=0, column=0, padx=10, pady=10)

        self.entry_x = ctk.CTkEntry(self.main_frame)
        self.entry_x.grid(row=0, column=1, padx=10, pady=10)

        self.label_y = ctk.CTkLabel(self.main_frame, text="Result y:")
        self.label_y.grid(row=1, column=0, padx=10, pady=10)

        self.result_label = ctk.CTkLabel(self.main_frame, text="---")
        self.result_label.grid(row=1, column=1, padx=10, pady=10)

        self.plot_frame = ctk.CTkFrame(self, height=400)
        self.plot_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

    def train_model(self):
        try:
            df = pd.read_csv(self.csv_file)

            x = df['x'].values.reshape(-1, 1)
            y = df['y'].values.reshape(-1, 1)

            model = LinearRegression()
            model.fit(x, y)

            pickle.dump(model, open(self.model_file, 'wb'))

            self.plot_model(x, y, model)

            messagebox.showinfo("Success", "Model trained and saved successfully!")
        except FileNotFoundError:
            messagebox.showerror("Error", f"CSV file '{self.csv_file}' not found.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def predict_value(self):
        try:
            x = self.entry_x.get()

            if not x:
                messagebox.showwarning("Input Error", "Please enter a value for x.")
                return

            x = float(x)

            model = pickle.load(open(self.model_file, 'rb'))

            x_value = np.array([x]).reshape(-1, 1)
            y_predicted = model.predict(x_value)[0][0]

            self.result_label.configure(text=f"{y_predicted:.2f}")

            self.update_model(x, y_predicted)
        except FileNotFoundError:
            messagebox.showerror("Error", f"Model file '{self.model_file}' not found. Please train the model first.")
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid numeric value for x.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def update_model(self, new_x, new_y):
        try:
            new_y = round(new_y, 2)

            df = pd.read_csv(self.csv_file)
            df.loc[len(df.index)] = [new_x, new_y]
            df.to_csv(self.csv_file, index=False)

            x = df['x'].values.reshape(-1, 1)
            y = df['y'].values.reshape(-1, 1)

            model = LinearRegression()
            model.fit(x, y)

            pickle.dump(model, open(self.model_file, 'wb'))

            self.plot_model(x, y, model)
        except FileNotFoundError:
            messagebox.showerror("Error", f"CSV file '{self.csv_file}' not found.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def plot_model(self, x, y, model):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)

        ax.scatter(x, y, color='blue', label='Dane')
        ax.plot(x, model.predict(x), color='red', label='Model')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Model Linear Regression')
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


if __name__ == "__main__":
    app = LinearRegressionApp()
    app.mainloop()
