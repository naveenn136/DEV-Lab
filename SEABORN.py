import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Load the dataset
titanic = sns.load_dataset("titanic")

# -------------------- 1. Data Cleaning --------------------
# Fill missing age with median
titanic['age'].fillna(titanic['age'].median(), inplace=True)

# Fill missing embarked with mode
titanic['embarked'].fillna(titanic['embarked'].mode()[0], inplace=True)

# Drop 'deck' due to many missing values
titanic.drop(columns='deck', inplace=True)

# Drop any remaining rows with nulls
titanic.dropna(inplace=True)

# -------------------- 2. Interactive Survival Count --------------------
fig1 = px.histogram(titanic, x="survived", color="survived",
                    color_discrete_map={0: 'red', 1: 'green'},
                    title="Survival Count",
                    labels={"survived": "Survived"},
                    category_orders={"survived": [0, 1]})
fig1.update_layout(xaxis=dict(tickvals=[0, 1], ticktext=["No", "Yes"]))
fig1.show()

# -------------------- 3. Survival by Sex --------------------
fig2 = px.histogram(titanic, x="sex", color="survived",
                    barmode="group",
                    title="Survival Count by Gender",
                    labels={"survived": "Survived"})
fig2.show()

# -------------------- 4. Survival by Passenger Class --------------------
fig3 = px.histogram(titanic, x="pclass", color="survived",
                    barmode="group",
                    title="Survival Count by Passenger Class",
                    labels={"pclass": "Passenger Class", "survived": "Survived"})
fig3.show()

# -------------------- 5. Age Distribution --------------------
fig4 = px.histogram(titanic, x="age", nbins=30, marginal="box",
                    title="Age Distribution of Passengers")
fig4.show()

# -------------------- 6. Age vs Fare Scatter Plot --------------------
fig5 = px.scatter(titanic, x="age", y="fare", color="survived",
                  symbol="sex", title="Age vs Fare (Colored by Survival)",
                  labels={"survived": "Survived"})
fig5.show()

# -------------------- 7. Correlation Heatmap --------------------
# Encode categorical for correlation
corr_data = titanic.copy()
corr_data['sex'] = corr_data['sex'].map({'male': 0, 'female': 1})
corr_data['embarked'] = corr_data['embarked'].map({'S': 0, 'C': 1, 'Q': 2})

corr_matrix = corr_data.corr(numeric_only=True)

fig6 = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,
    x=corr_matrix.columns,
    y=corr_matrix.columns,
    colorscale='Viridis'
))
fig6.update_layout(title="Correlation Heatmap")
fig6.show()

# -------------------- 8. Survival by Embarkation Port --------------------
fig7 = px.histogram(titanic, x="embarked", color="survived",
                    barmode="group", title="Survival by Embarkation Port")
fig7.show()
