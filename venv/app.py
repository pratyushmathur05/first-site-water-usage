from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
import random
from datetime import datetime, timedelta
import os 
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
app = Flask(__name__)


def simulate_water_usage():
    return {
        "cooking": random.randint(5, 20),
        "bathing": random.randint(20, 50),
        "laundry": random.randint(10, 30),
        "cleaning": random.randint(5, 15),
        "drinking": random.randint(1, 5)
    }

def visualize_bar_chart(df):
    category_totals = df[['cooking', 'bathing', 'laundry', 'cleaning', 'drinking']].sum()

    plt.figure(figsize=(8, 6))
    category_totals.plot(kind='bar', color=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0'])
    plt.title('Total Water Usage by Category')
    plt.xlabel('Category')
    plt.ylabel('Total Water Usage (liters)')
    plt.xticks(rotation=45)

    bar_chart_path = os.path.join('static', 'bar_chart.png')
    plt.tight_layout()
    plt.savefig(bar_chart_path)
    print(f"Bar chart saved to {bar_chart_path}")
    plt.clf()  

    return '/static/bar_chart.png'


def generate_usage_data(num_days=30):
    start_date = datetime.now() - timedelta(days=num_days)
    data = []

    for i in range(num_days):
        date = start_date + timedelta(days=i)
        usage = simulate_water_usage()
        data.append({"date": date, "cooking": usage["cooking"], "bathing": usage["bathing"],
                     "laundry": usage["laundry"], "cleaning": usage["cleaning"], "drinking": usage["drinking"]})
    
    df = pd.DataFrame(data)
    print(df.head())  # Check the DataFrame structure here as well
    return df

# Visualize water usage breakdown (stackplot)
def visualize_usage(df):
    plt.figure(figsize=(10, 6))
    plt.stackplot(df['date'], df['cooking'], df['bathing'], df['laundry'], df['cleaning'], df['drinking'],
                  labels=["Cooking", "Bathing", "Laundry", "Cleaning", "Drinking"], alpha=0.6)
    plt.title('Water Usage Breakdown Over Time')
    plt.xlabel('Date')
    plt.ylabel('Water Usage (liters)')
    plt.xticks(rotation=45)
    plt.legend(loc='upper left')
    plt.grid(True)
    graph_path = os.path.join('static', 'usage_graph.png')
    plt.tight_layout()
    plt.savefig(graph_path)
    plt.close()

def visualize_line_graph(df):
    plt.figure(figsize=(10, 6))

    # Plot each activity's data
    for activity in ['cooking', 'bathing', 'laundry', 'cleaning', 'drinking']:
        plt.plot(df['date'], df[activity], label=activity.capitalize())
    
    plt.title('Water Usage Trend Over Time')
    plt.xlabel('Date')
    plt.ylabel('Water Usage (liters)')
    plt.xticks(rotation=45)
    plt.legend(loc='upper left')
    plt.grid(True)

    # Save the line graph
    line_graph_path = os.path.join('static', 'line_graph.png')
    plt.tight_layout()
    plt.savefig(line_graph_path)
    plt.close()

    return '/static/line_graph.png'

def visualize_pie_chart(df):
    # Summing the usage across all days for each activity
    total_usage = df[['cooking', 'bathing', 'laundry', 'cleaning', 'drinking']].sum()

    # Create the pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(total_usage, labels=total_usage.index, autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0'])
    plt.title('Total Water Usage Breakdown')

    # Save the pie chart
    pie_chart_path = os.path.join('static', 'pie_chart.png')
    plt.tight_layout()
    plt.savefig(pie_chart_path)
    plt.close()

    return '/static/pie_chart.png'

def analyze_usage(df):
    df_without_date = df.drop(columns=["date"])
    total_usage = df_without_date.sum(axis=0)
    avg_usage = df_without_date.mean(axis=0)
    return total_usage, avg_usage

# Manage users (using Flask sessions)
def get_current_user():
    return session.get('username', None)

# # Define the explanation generation function using Gamini
# def generate_explanation(df):
#     # Prepare the data for the prompt
#     total_usage, avg_usage = analyze_usage(df)

#     # Format the usage data into a string to be passed to the model
#     total_usage_str = "\n".join([f"{key}: {value} liters" for key, value in total_usage.items()])
#     avg_usage_str = "\n".join([f"{key}: {value:.2f} liters/day" for key, value in avg_usage.items()])

#     # Create the prompt for Gamini
#     prompt = f"""
#     Here is the water usage data for the past 30 days:

#     Total water usage:
#     {total_usage_str}

#     Average water usage per day:
#     {avg_usage_str}

#     Based on the above data, provide a detailed explanation in at least three paragraphs about the water usage trends and possible reasons for any patterns observed.
#     """

#     # Get the response from the Gamini model
#     response = Completion.create(
#         api_key="YOUR_GAMINI_API_KEY",  # Replace with your Gamini API key
#         prompt=prompt,
#         max_tokens=300,
#         temperature=0.7
#     )

    # Return the explanation
    return response["choices"][0]["text"].strip()

# @app.route('/generate_explanation', methods=['GET'])
# def generate_data_explanation():
#     df = generate_usage_data()
#     explanation = generate_explanation(df)

#     return render_template('explanation.html', explanation=explanation)


@app.route('/', methods=['GET', 'POST'])
def home():
    # Load or generate data for the last 30 days
    df = generate_usage_data()

    if request.method == 'POST':
        # Get user input for today's water usage
        date = datetime.now().strftime('%Y-%m-%d')
        new_data = {
            "date": date,
            "cooking": request.form.get('cooking', type=int),
            "bathing": request.form.get('bathing', type=int),
            "laundry": request.form.get('laundry', type=int),
            "cleaning": request.form.get('cleaning', type=int),
            "drinking": request.form.get('drinking', type=int)
        }

        # Ensure that 'date' column is in datetime format and handle any errors
        df['date'] = pd.to_datetime(df['date'], errors='coerce')  # Convert to datetime, invalid values become NaT

        # Drop rows where 'date' is NaT (invalid dates)
        df = df.dropna(subset=['date'])

        # Now, drop duplicates and sort by date
        df = df.drop_duplicates(subset='date').sort_values(by='date').reset_index(drop=True)

        # Check if data for today exists; if yes, update it
        if date in df['date'].values:
            df.loc[df['date'] == date, ['cooking', 'bathing', 'laundry', 'cleaning', 'drinking']] = \
                [new_data['cooking'], new_data['bathing'], new_data['laundry'], new_data['cleaning'], new_data['drinking']]
        else:
            # Add today's data if it doesn't exist
            df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)

        # Visualize data
        pie_chart_url = visualize_pie_chart(df)
        line_graph_url = visualize_line_graph(df)
        visualize_usage(df)

        # Analyze usage
        total_usage, avg_usage = analyze_usage(df)

        return render_template(
            'index.html',
            data=df.to_html(classes="table table-bordered"),
            total_usage=total_usage,
            avg_usage=avg_usage,
            graph_url="/static/usage_graph.png",
            pie_chart_url=pie_chart_url,
            line_graph_url=line_graph_url
        )

    # Default behavior (GET request)
    visualize_usage(df)
    pie_chart_url = visualize_pie_chart(df)
    line_graph_url = visualize_line_graph(df)
    total_usage, avg_usage = analyze_usage(df)

    return render_template(
        'index.html',
        data=df.to_html(classes="table table-bordered"),
        total_usage=total_usage,
        avg_usage=avg_usage,
        graph_url="/static/usage_graph.png",
        pie_chart_url=pie_chart_url,
        line_graph_url=line_graph_url
    )

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        session['username'] = username
        return redirect(url_for('home'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('data', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
