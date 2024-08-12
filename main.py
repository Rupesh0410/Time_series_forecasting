import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from io import BytesIO
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from PIL import Image
from bs4 import BeautifulSoup
app = FastAPI()

# Set up templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the trained model and scaler
output_dir = 'model'
model_path = os.path.join(output_dir, 'trained_model.h5')
model = load_model(model_path)
scaler = MinMaxScaler()

# Load the scaler and test data
train_data_path = os.path.join(output_dir, 'train_data.csv')
test_data_path = os.path.join(output_dir, 'test_data.csv')
train_data = pd.read_csv(train_data_path, index_col='DATE')
scaler.fit(train_data)
test_data = pd.read_csv(test_data_path, index_col='DATE')
scaled_test = scaler.transform(test_data)

# Function to generate predictions
def generate_predictions(model, scaler, scaled_test, n_input, num_months, output_dir):
    predictions = []
    input_seq = scaled_test[-n_input:]

    for _ in range(num_months):
        input_seq = input_seq.reshape((1, n_input, 1))
        next_pred = model.predict(input_seq)[0]
        predictions.append(next_pred)
        input_seq = np.append(input_seq[:, 1:, :], [[next_pred]], axis=1)

    predictions = scaler.inverse_transform(predictions)
    last_date = pd.to_datetime(test_data.index[-1])
    date_range = pd.date_range(last_date, periods=num_months + 1, freq='MS')[1:]
    predictions_df = pd.DataFrame(predictions, index=date_range, columns=['Predicted_Sales'])

    # Calculate difference and percentage change
    predictions_df['Difference'] = predictions_df['Predicted_Sales'].diff()
    predictions_df['Percentage Change'] = predictions_df['Difference'] / predictions_df['Predicted_Sales'] * 100

    # Format the Percentage Change column with arrows
    def format_percentage_change(value):
        if pd.isna(value):
            return ''
        elif value > 0:
            return f'<span style="color: green;">⬆️ {abs(value):.2f}%</span>'
        elif value < 0:
            return f'<span style="color: red;">⬇️ {abs(value):.2f}%</span>'
        else:
            return f'{abs(value):.2f}%'

    predictions_df['Percentage Change'] = predictions_df['Percentage Change'].apply(format_percentage_change)

    # Save predictions to CSV
    output_file = os.path.join(output_dir, 'forecast_table.csv')
    predictions_df.to_csv(output_file, index=True)
    
    return predictions_df

# Function to get predicted value for a specific month
def get_predicted_value(predictions_df, specific_date):
    try:
        specific_date = pd.to_datetime(specific_date)
        if specific_date in predictions_df.index:
            predicted_value = predictions_df.loc[specific_date, 'Predicted_Sales']
            return predicted_value
        else:
            return None
    except ValueError:
        return None

# Function to plot test vs predicted data
def plot_test_vs_pred(test_data, predictions_df, width=10, height=6):
    test_dates = pd.to_datetime(test_data.index)  # Convert index to datetime if not already
    predictions_dates = pd.to_datetime(predictions_df.index)  # Convert index to datetime if not already

    plt.figure(figsize=(width, height))
    plt.plot(test_dates, test_data.values, label="Actual Sales")
    plt.plot(predictions_dates, predictions_df['Predicted_Sales'], label="Predicted Sales", linestyle='dashed')
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.title("Previous vs Predicted Sales")
    plt.legend()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

def format_predictions_html(predictions_html):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(predictions_html, "html.parser")
    table = soup.find("table")
    rows = table.find_all("tr")[1:]  # Skip header row

    for row in rows:
        cols = row.find_all("td")
        if len(cols) >= 4:  # Ensure there are enough columns
            diff = float(cols[2].text)
            perc_change = float(cols[3].text.strip('%'))  # Strip '%' sign and convert to float

            # Format the percentage change with arrow
            if perc_change > 0:
                arrow = '⬆️'
                color = 'green'
            elif perc_change < 0:
                arrow = '⬇️'
                color = 'red'
            else:
                arrow = ''
                color = 'black'

            cols[3].string = f'<span style="color: {color};">{arrow} {abs(perc_change):.2f}%</span>'

    return str(soup)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # Load test_vs_predictions.csv
    test_vs_predictions_path = os.path.join(output_dir, 'test_vs_predictions.csv')
    if (test_vs_predictions_path):
        test_vs_predictions_df = pd.read_csv(test_vs_predictions_path)
        test_vs_predictions_html = test_vs_predictions_df.to_html(classes='table table-striped', index=False)
    else:
        test_vs_predictions_html = None

    return templates.TemplateResponse("index.html", {"request": request, "test_vs_predictions": test_vs_predictions_html})

@app.post("/get_prediction")
async def get_prediction(request: Request, num_months: int = Form(...)):
    global global_num_months
    global_num_months = num_months  # Store the number of months globally
    
    n_input = min(12, len(scaled_test) - 1)
    predictions_df = generate_predictions(model, scaler, scaled_test, n_input, num_months, output_dir)
    predictions_df.columns.name = 'Date'

    # Convert the dataframe to HTML with custom formatting for the difference and percentage change columns
    predictions_html = predictions_df.to_html(classes='table table-striped', index=True, escape=False)
    predictions_html = format_predictions_html(predictions_html)

    plot_buf = plot_test_vs_pred(test_data, predictions_df)
    plot_path = os.path.join(output_dir, "sales_vs_predictions.png")
    with open(plot_path, 'wb') as f:
        f.write(plot_buf.getbuffer())
    
    return templates.TemplateResponse("index.html", {"request": request, "predictions": predictions_html, "plot_path": plot_path, "num_months": num_months})

@app.post("/get_specific_prediction")
async def get_specific_prediction(request: Request, specific_date: str = Form(...)):
    global global_num_months
    
    n_input = min(12, len(scaled_test) - 1)
    predictions_df = generate_predictions(model, scaler, scaled_test, n_input, num_months=global_num_months, output_dir=output_dir)
    predicted_value = get_predicted_value(predictions_df, specific_date)

    # Convert the dataframe to HTML with custom formatting for the difference and percentage change columns
    predictions_html = predictions_df.to_html(classes='table table-striped', index=True, escape=False)
    predictions_html = format_predictions_html(predictions_html)
    
    plot_buf = plot_test_vs_pred(test_data, predictions_df)
    plot_path = os.path.join(output_dir, "sales_vs_predictions.png")
    with open(plot_path, 'wb') as f:
        f.write(plot_buf.getbuffer())

    if predicted_value is not None:
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "specific_date": specific_date, 
            "predicted_value": predicted_value,
            "predictions": predictions_html,
            "plot_path": plot_path
        })
    else:
        error_message = f"Prediction not found for {specific_date}."
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "error_message": error_message,
            "predictions": predictions_html,
            "plot_path": plot_path
        })

@app.post("/get_predictions_range")
async def get_predictions_range(request: Request, start_date: str = Form(...), end_date: str = Form(...)):
    global global_num_months
    
    n_input = min(12, len(scaled_test) - 1)
    predictions_df = generate_predictions(model, scaler, scaled_test, n_input, num_months=global_num_months, output_dir=output_dir)
    
    try:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        predictions_range_df = predictions_df[start_date:end_date]
        
        if predictions_range_df.empty:
            error_message = f"No predictions available for the specified range {start_date} to {end_date}."
            return templates.TemplateResponse("index.html", {
                "request": request, 
                "error_message": error_message
            })
        
        predictions_html = predictions_range_df.to_html(classes='table table-striped', index=True, escape=False)
        predictions_html = format_predictions_html(predictions_html)
        
        plot_buf = plot_test_vs_pred(test_data, predictions_df)
        plot_path = os.path.join(output_dir, "sales_vs_predictions.png")
        with open(plot_path, 'wb') as f:
            f.write(plot_buf.getbuffer())
        
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "predictions_range": predictions_html,
            "plot_path": plot_path,
            "start_date": start_date,
            "end_date": end_date
        })
    except ValueError:
        error_message = "Invalid date format. Please use YYYY-MM-DD."
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "error_message": error_message
        })
    
@app.get("/plot", response_class=FileResponse)
async def get_plot():
    plot_path = os.path.join(output_dir, "sales_vs_predictions.png")

    # Resize the image
    with Image.open(plot_path) as img:
        resized_img = img.resize((1100, 660))  # Set your desired width and height
        resized_img_path = os.path.join(output_dir, "resized_sales_vs_predictions.png")
        resized_img.save(resized_img_path)

    return FileResponse(resized_img_path)

@app.get("/graph", response_class=FileResponse)
async def get_graph():
    graph_path = os.path.join(output_dir, "test_graph.png")

    # Resize the image
    with Image.open(graph_path) as img:
        resized_img = img.resize((1100, 660))  # Set your desired width and height
        resized_img_path = os.path.join(output_dir, "resized_test_graph.png")
        resized_img.save(resized_img_path)

    return FileResponse(resized_img_path)

@app.get("/training_validation_graph", response_class=FileResponse)
async def get_training_validation_graph():
    graph_path = os.path.join(output_dir, "training_validation_loss.png")

    # Resize the image
    with Image.open(graph_path) as img:
        resized_img = img.resize((1100, 660))  # Set your desired width and height
        resized_img_path = os.path.join(output_dir, "resized_training_validation_loss.png")
        resized_img.save(resized_img_path)

    return FileResponse(resized_img_path)

@app.get("/graphs", response_class=HTMLResponse)
async def read_graphs(request: Request):
    return templates.TemplateResponse("graphs.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
