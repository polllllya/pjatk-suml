import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import matplotlib.pyplot as plt


def predict_value(x, model_file='our_model.pkl'):
    imported_model = pickle.load(open(model_file, 'rb'))

    x_unknown = np.array([x]).reshape(-1, 1)
    y_unknown = imported_model.predict(x_unknown)

    return y_unknown[0][0]


def update_model(new_x, new_y, csv_file='10_points.csv', model_file='our_model.pkl'):
    df = pd.read_csv(csv_file)

    df.loc[len(df.index)] = [new_x, new_y]

    df.to_csv(csv_file, index=False)

    x = df['x'].values.reshape(-1, 1)
    y = df['y'].values.reshape(-1, 1)

    our_model = LinearRegression()
    our_model.fit(x, y)

    pickle.dump(our_model, open(model_file, 'wb'))
    print('Model został zaktualizowany i zapisany w pliku', model_file)

    plt.scatter(x, y, color='blue', label='dane')
    plt.plot(x, our_model.predict(x), color='red', label='new_model')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('update model')
    plt.legend()
    plt.show()


def main():
    test_x = 2.78
    new_x, new_y = 11, 21.65

    try:
        predicted_y = predict_value(test_x)
        print(f"Przewidywana wartość y dla x={test_x}: {predicted_y}")
    except FileNotFoundError:
        print("Plik modelu nie istnieje")

    print("\n=== Aktualizacja modelu nowymi danymi ===")
    try:
        update_model(new_x, new_y)
        print(f"Model został zaktualizowany nowymi danymi: x={new_x}, y={new_y}")
    except FileNotFoundError:
        print("Plik nie istnieje")

    print("\n=== Sprawdzanie przewidywania po aktualizacji ===")
    try:
        predicted_y_after_update = predict_value(test_x)
        print(f"Nowa przewidywana wartość y dla x={test_x}: {predicted_y_after_update}")
    except FileNotFoundError:
        print("Plik modelu nie istnieje")


if __name__ == "__main__":
    main()

