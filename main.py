from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Загружаем датасет
file_path = 'kc_house_data.csv'
data = pd.read_csv(file_path)

# Обробка пропущених значень
data = data.dropna()

# Масштабування числових даних
scaler = StandardScaler()
scaled_features = ['sqft_living', 'price', 'grade', 'condition']
data[scaled_features] = scaler.fit_transform(data[scaled_features])

# Розділення даних для навчання моделі
X = data[['sqft_living', 'grade', 'condition']]
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Перетворення в тензори для PyTorch
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).cuda()
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).cuda()
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).cuda()
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).cuda()


# Визначення моделі для прогнозування ціни
class HousePricePredictor(nn.Module):
    def __init__(self, input_dim):
        super(HousePricePredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(x)


# Ініціалізація моделі для ціни
price_model = HousePricePredictor(input_dim=X_train.shape[1]).cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(price_model.parameters(), lr=0.001)


# Навчання моделі
def train_model(model, X_train, y_train, criterion, optimizer, epochs=50):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')


train_model(price_model, X_train_tensor, y_train_tensor, criterion, optimizer)


# Функція рекомендації будинків
def get_recommendations(input_data):
    """
    Рекомендує будинки та розраховує окупність і перспективи росту.

    Параметри:
    - input_data: dict, масив даних у форматі:
        {
            "desired_sqft": int,   # Бажана площа будинку
            "max_budget": int,    # Максимальний бюджет
            "price_growth_rate": float,         # Очікуваний річний ріст вартості (у відсотках)
            "desired_rental_income": float      # Бажаний дохід від оренди
        }

    Повертає:
    - list, список рекомендованих будинків з розрахунками.
    """
    # Розпакування параметрів
    desired_sqft = input_data["desired_sqft"]
    max_budget = input_data["max_budget"]
    price_growth_rate = input_data["price_growth_rate"]
    desired_rental_income = input_data["desired_rental_income"]

    # Масштабування введених даних
    scaled_input = scaler.transform([[desired_sqft, max_budget, 0, 0]])[0]
    user_tensor = torch.tensor([[scaled_input[0], 0, 0]], dtype=torch.float32).cuda()

    price_model.eval()
    with torch.no_grad():
        predicted_price_scaled = price_model(user_tensor).item()

    # Повернення до оригінальної шкали
    predicted_price_original = scaler.inverse_transform([[0, predicted_price_scaled, 0, 0]])[0][1]

    # Прогнозування ренти, залежно від ціни житла
    optimal_rental_income = 0.01 * predicted_price_original  # Припустимо, рента становить 1% від ціни житла на місяць

    if optimal_rental_income < desired_rental_income:
        rental_income_suggestion = optimal_rental_income
    else:
        rental_income_suggestion = desired_rental_income

    # Фільтруємо дані, які відповідають критеріям
    filtered_data = data[(data['sqft_living'] * scaler.scale_[0] + scaler.mean_[0] >= desired_sqft) &
                         (data['price'] * scaler.scale_[1] + scaler.mean_[1] <= max_budget)]

    if filtered_data.empty:
        return {"error": "Немає будинків, які відповідають критеріям"}

    recommended = filtered_data.sort_values(by=['grade', 'condition', 'sqft_living'], ascending=False).head(5)

    # Розрахунок окупності
    recommended['price_original'] = recommended['price'] * scaler.scale_[1] + scaler.mean_[1]
    recommended['annual_rental_income'] = rental_income_suggestion * 12
    recommended['roi_years'] = recommended['price_original'] / recommended['annual_rental_income']
    recommended['projected_price_growth'] = recommended['price_original'] * (1 + price_growth_rate) ** 5

    # Додаємо рекомендовану ренту як окремий стовпчик
    recommended['recommended_rental_income'] = rental_income_suggestion

    return recommended[['id', 'price_original', 'annual_rental_income', 'recommended_rental_income', 'roi_years',
                        'projected_price_growth']].to_dict('records')


# Приклад використання
input_data = {
    "desired_sqft": 2000,  # Бажана площа
    "max_budget": 500000,  # Максимальний бюджет
    "price_growth_rate": 0.05,  # Річний ріст вартості
    "desired_rental_income": 2500  # Бажаний дохід від оренди
}


# Функція для розрахунку рекомендацій (заглушка для прикладу)
#ef get_recommendations(input_data):
#   # Тут ти можеш вставити логіку обробки даних або моделі
#   # Для прикладу створюємо фейкові дані
#   recommended = [
#       {
#           "id": 1,
#           "price_original": 450000,
#           "annual_rental_income": 27000,
#           "recommended_rental_income": 2500,
#           "roi_years": 15,
#           "projected_price_growth": 5.5
#       },
#       {
#           "id": 2,
#           "price_original": 480000,
#           "annual_rental_income": 29000,
#           "recommended_rental_income": 2600,
#           "roi_years": 16,
#           "projected_price_growth": 4.8
#       }
#   ]
#   return recommended


# Рендер сторінки для введення даних
@app.route('/')
def index():
    return render_template('index.html')


# Обробка POST-запиту та повернення результату
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Приймаємо JSON-дані від клієнта
        data = request.get_json()
        print("Received Input Data:", data)

        # Викликаємо функцію для отримання рекомендацій
        recommendations = get_recommendations(data)

        # Повертаємо результати у форматі JSON
        return jsonify(recommendations), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
