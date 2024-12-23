from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import boto3
import pandas as pd
from io import StringIO
# Загружаем датасет

BUCKET_NAME = "databigdataproject"
FILE_KEY = "kc_house_data.csv(1)/kc_house_data.csv"
AWS_REGION = "eu-north-1"

def load_csv_from_s3(bucket_name, file_key, region):

   
        # Create an S3 client
        s3 = boto3.client('s3', region_name=region)
        
        # Fetch the object from S3
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        
        # Read the CSV data
        csv_data = response['Body'].read().decode('utf-8')
        
        # Load the data into a Pandas DataFrame
        df = pd.read_csv(StringIO(csv_data))
        return df
    
   


    
data = load_csv_from_s3(BUCKET_NAME, FILE_KEY, AWS_REGION)



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
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)  # Без використання .cuda()
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)  # Без використання .cuda()
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)  # Без використання .cuda()
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)  # Без використання .cuda()

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
price_model = HousePricePredictor(input_dim=X_train.shape[1])  # Без використання .cuda()
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

# Глобальна змінна price_growth_rate
price_growth_rate = 0.12  # Річний ріст вартості

# Функція рекомендації будинків
def get_recommendations(input_data):
    """
    Рекомендує будинки та розраховує окупність і перспективи росту.

    Параметри:
    - input_data: dict, масив даних у форматі:
        {
            "desired_sqft": int,   # Бажана площа будинку
            "max_budget": int,    # Максимальний бюджет
            "desired_rental_income": float      # Бажаний дохід від оренди
        }

    Повертає:
    - list, список рекомендованих будинків з розрахунками.
    """
    # Розпакування параметрів
    desired_sqft = float(input_data["desired_sqft"])
    max_budget = float(input_data["max_budget"])
    desired_rental_income = float(input_data["desired_rental_income"])

    # Масштабування введених даних
    scaled_input = scaler.transform([[desired_sqft, max_budget, 0, 0]])[0]  # Масштабуємо вхідні дані
    user_tensor = torch.tensor([[scaled_input[0], scaled_input[1], scaled_input[2]]], dtype=torch.float32)  # Без .cuda()

    price_model.eval()
    with torch.no_grad():
        predicted_price_scaled = price_model(user_tensor).item()  # Отримуємо передбачену ціну

    # Повернення до оригінальної шкали (тільки ціни)
    predicted_price_original = scaler.inverse_transform([[0, predicted_price_scaled, 0, 0]])[0][1]

    # Переконаємось, що ціна не стала від'ємною
    predicted_price_original = max(predicted_price_original, 0)

    # Прогнозування ренти, залежно від ціни житла
    # Для рентабельності беремо 5% від ціни на місяць
    rental_income_suggestion = 0.01 * predicted_price_original  # 5% від ціни житла на місяць


    # Фільтруємо дані, які відповідають критеріям
    filtered_data = data[(data['sqft_living'] * scaler.scale_[0] + scaler.mean_[0] >= desired_sqft) & 
                         (data['price'] * scaler.scale_[1] + scaler.mean_[1] <= max_budget)]

    if filtered_data.empty:
        return {"error": "Немає будинків, які відповідають критеріям"}

    recommended = filtered_data.sort_values(by=['grade', 'condition', 'sqft_living'], ascending=False).head(5)

    # Розрахунок оригінальної ціни та ренти
    recommended['price_original'] = recommended['price'] * scaler.scale_[1] + scaler.mean_[1]
    recommended['annual_rental_income'] = rental_income_suggestion * 12  # Перетворюємо в річний дохід
    recommended['roi_years'] = recommended['price_original'] / recommended['annual_rental_income']  # Розрахунок ROI
    recommended['projected_price_growth'] = recommended['price_original'] * (1 + price_growth_rate) ** 5  # Прогноз росту

    # Переконаємось, що ціни та рента не від'ємні
    recommended['price_original'] = recommended['price_original'].apply(lambda x: max(x, 0))
    recommended['annual_rental_income'] = recommended['annual_rental_income'].apply(lambda x: max(x, 0))

    # Виправлення для ROI: якщо ROI є нескінченним (тобто annual_rental_income = 0), поставити значення 0 або інше коректне значення
    recommended['roi_years'] = recommended['roi_years'].apply(lambda x: x if x != float('inf') else 0)
    recommended['projected_price_growth'] = recommended['projected_price_growth'].apply(lambda x: max(x, 0))

    # Додаємо рекомендовану ренту як окремий стовпчик
    recommended['recommended_rental_income'] = rental_income_suggestion

    # Повертаємо дані у форматі, який можна серіалізувати в JSON
    return recommended[['id', 'price_original', 'annual_rental_income', 'recommended_rental_income', 'roi_years',
                        'projected_price_growth']].to_dict('records')

# Рендер сторінки для введення даних
@app.route('/')
def index():
    return render_template('index.html')

# Обробка POST-запиту та повернення результату
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Приймаємо JSON-дані від клієнта
        data_1 = request.get_json()
        print("Received Input Data:", data_1)

        # Викликаємо функцію для отримання рекомендацій
        recommendations = get_recommendations(data_1)

        # Повертаємо результати у форматі JSON
        return jsonify(recommendations), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=False, host = '0.0.0.0'  , port=20)
