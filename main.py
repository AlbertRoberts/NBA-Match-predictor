import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
from nba_api.stats.endpoints import leaguedashteamstats
from nba_api.stats.static import teams


# Function to get NBA team statistics for a specific season
def get_nba_team_stats(season):
    season_str = f"{season}-{str(season + 1)[-2:]}"

    # Get regular season team stats
    team_stats = leaguedashteamstats.LeagueDashTeamStats(
        measure_type_detailed_defense='Base',
        per_mode_detailed='PerGame',
        season=season_str,
        season_type_all_star='Regular Season'
    )

    # Convert to DataFrame
    stats_df = team_stats.get_data_frames()[0]

    # Select relevant columns
    selected_columns = [
        'TEAM_NAME', 'W', 'L', 'FG_PCT', 'FG3_PCT', 'FT_PCT',
        'REB', 'AST', 'TOV', 'STL', 'BLK', 'PTS', 'PLUS_MINUS'
    ]

    stats_df = stats_df[selected_columns].copy()

    # Add season column
    stats_df['SEASON'] = season

    return stats_df


# Collect data for multiple seasons (2013-2023)
all_seasons_data = []
for season in range(2013, 2023):
    try:
        print(f"Fetching data for season {season}-{season + 1}...")
        season_data = get_nba_team_stats(season)
        all_seasons_data.append(season_data)
    except Exception as e:
        print(f"Error fetching data for season {season}: {e}")

# Combine all seasons into one DataFrame
nba_data = pd.concat(all_seasons_data, ignore_index=True)
print(f"Collected data for {len(nba_data)} team-seasons")

# Display the first few rows of the dataset
print("\nNBA Dataset Preview:")
print(nba_data.head())

# Data preprocessing
# Rename columns for clarity
nba_data = nba_data.rename(columns={
    'W': 'Wins',
    'L': 'Losses',
    'FG_PCT': 'FG_PCT',
    'FG3_PCT': 'THREE_PT_PCT',
    'FT_PCT': 'FT_PCT',
    'REB': 'Rebounds',
    'AST': 'Assists',
    'TOV': 'Turnovers',
    'STL': 'Steals',
    'BLK': 'Blocks',
    'PTS': 'Points_Scored',
    'PLUS_MINUS': 'Point_Differential',
    'TEAM_NAME': 'Team',
    'SEASON': 'Season'
})

# Calculate Points_Allowed from Points_Scored and Point_Differential
nba_data['Points_Allowed'] = nba_data['Points_Scored'] - nba_data['Point_Differential']

# Split data: use the last season for testing, earlier seasons for training
test_season = 2022
train_data = nba_data[nba_data['Season'] < test_season]
test_data = nba_data[nba_data['Season'] == test_season]

print(f"\nTraining data: {len(train_data)} team-seasons")
print(f"Test data: {len(test_data)} team-seasons")

# Define features and target variable
features = [
    'FG_PCT', 'THREE_PT_PCT', 'FT_PCT', 'Rebounds', 'Assists',
    'Steals', 'Blocks', 'Turnovers', 'Points_Scored', 'Points_Allowed',
    'Point_Differential'
]

X_train = train_data[features]
y_train = train_data['Wins']
X_test = test_data[features]
y_test = test_data['Wins']

# Feature selection using SelectKBest
# Select the 5 most important features
k_best = SelectKBest(score_func=f_regression, k=5)
X_train_best = k_best.fit_transform(X_train, y_train)
X_test_best = k_best.transform(X_test)

# Get the names of the selected features
selected_features_mask = k_best.get_support()
selected_features = [features[i] for i in range(len(features)) if selected_features_mask[i]]
print("\nSelected Features:")
print(selected_features)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train_best, y_train)

# Make predictions
y_pred_train = model.predict(X_train_best)
y_pred_test = model.predict(X_test_best)


# Evaluate the model
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(1, y_true))) * 100  # Mean Absolute Percentage Error
    accuracy = 100 - mape  # Percentage accuracy

    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")
    print(f"Prediction Accuracy: {accuracy:.2f}%")

    return mae, rmse, r2, accuracy


print("\nTraining Set Performance:")
train_metrics = evaluate_model(y_train.values, y_pred_train)

print("\nTest Set Performance:")
test_metrics = evaluate_model(y_test.values, y_pred_test)

# Visualize prediction accuracy with a scatter plot
plt.figure(figsize=(12, 8))
plt.scatter(y_test, y_pred_test, alpha=0.7, s=80)
plt.plot([0, 82], [0, 82], 'r--', linewidth=2)  # Perfect prediction line

# Add team labels to points
for i, team in enumerate(test_data['Team']):
    plt.annotate(team, (y_test.iloc[i], y_pred_test[i]), fontsize=9)

plt.xlabel('Actual Wins', fontsize=12)
plt.ylabel('Predicted Wins', fontsize=12)
plt.title('NBA Win Prediction: Actual vs Predicted', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('nba_prediction_accuracy.png', dpi=300)
plt.show()

# Calculate residuals
test_data['Predicted_Wins'] = y_pred_test
test_data['Residuals'] = test_data['Wins'] - test_data['Predicted_Wins']

# Plot residuals
plt.figure(figsize=(12, 6))
sns.barplot(x='Team', y='Residuals', data=test_data.sort_values('Residuals'))
plt.axhline(y=0, color='r', linestyle='-')
plt.title('Prediction Residuals by Team', fontsize=14)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('nba_prediction_residuals.png', dpi=300)
plt.show()

# Feature importance analysis
if hasattr(model, 'coef_'):
    coefficients = pd.DataFrame({
        'Feature': selected_features,
        'Coefficient': model.coef_
    })
    coefficients = coefficients.sort_values(by='Coefficient', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Coefficient', y='Feature', data=coefficients)
    plt.title('Feature Importance for NBA Win Prediction', fontsize=14)
    plt.tight_layout()
    plt.savefig('nba_feature_importance.png', dpi=300)
    plt.show()

    print("\nFeature Importance:")
    print(coefficients)

# Create a comprehensive report
print("\n=== NBA Season Prediction Analysis Summary ===")
print(
    f"Data: {len(nba_data)} records for {nba_data['Team'].nunique()} teams over {nba_data['Season'].nunique()} seasons")
print(f"Model: Linear Regression with {len(selected_features)} selected features")
print(f"Features used: {', '.join(selected_features)}")
print(f"Prediction Accuracy: {test_metrics[3]:.1f}%")

# Save prediction results to CSV
results_df = test_data[['Team', 'Wins', 'Predicted_Wins', 'Residuals']].copy()
results_df['Predicted_Wins'] = results_df['Predicted_Wins'].round(1)
results_df['Absolute_Error'] = abs(results_df['Residuals'])
results_df = results_df.sort_values('Absolute_Error')
results_df.to_csv('nba_prediction_results.csv', index=False)
print(f"Prediction results saved to 'nba_prediction_results.csv'")
print("==============================================")


# Create a function for predicting future seasons
def predict_next_season(current_season_data, selected_features, model):
    """
    Uses the trained model to predict wins for the next season
    based on current season statistics with some adjustments for team changes.

    This is a simplified approach - real prediction would need to account for:
    - Roster changes
    - Player development/aging
    - Coaching changes
    - Schedule difficulty
    """
    # Make a copy of the current data
    next_season = current_season_data[['Team'] + selected_features].copy()

    # Predict wins using the model
    next_season_features = next_season[selected_features]
    predicted_wins = model.predict(next_season_features)

    # Create results dataframe
    next_season_predictions = pd.DataFrame({
        'Team': next_season['Team'],
        'Predicted_Wins': np.round(predicted_wins, 1)
    })

    return next_season_predictions.sort_values('Predicted_Wins', ascending=False)


# If we have data for the most recent season, predict the next one
if test_season == 2022:
    print("\n--- Predicted Standings for 2023-24 Season ---")
    next_season_predictions = predict_next_season(test_data, selected_features, model)
    print(next_season_predictions)
    next_season_predictions.to_csv('next_season_predictions.csv', index=False)