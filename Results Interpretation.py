import pandas as pd
from Data.LoadingData import DataLoader
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def get_results_df():
    results_df = pd.read_csv('FULL MOVE WITH PIECE DATA.csv')
    return results_df

def get_data():
    loader = DataLoader('VERSION 1/Data/lichess_db_puzzle.csv.zst')
    df = loader.backrank_and_m2()
    return df

def get_second_df():
    results_df = get_results_df()
    second_df = results_df
    return second_df

def get_fourth_df():
    results_df = get_results_df()
    # results_df = results_df[['GameUrl', 'correct_fourth_move', 'Rating']]
    return results_df

loader = DataLoader('VERSION 1/Data/lichess_db_puzzle.csv.zst')

# Load data
df = get_data()
results_df = get_results_df()
second_df = get_second_df()
fourth_df = get_fourth_df()


### PLOT OF ACCURACY VS RATING FULL MOVE ###

def plot_second_predictions(df):

    # Define rating ranges with steps of 100
    bins = list(range(0, 2600, 100))  # From 0 to 2500 in steps of 100
    labels = [f'{i}-{i+99}' for i in range(0, 2500, 100)]
    
    # Add a column for the rating range
    df = df.copy()
    df['RatingRange'] = pd.cut(df['Rating'], bins=bins, labels=labels, right=False, include_lowest=True)
    
    # Group by RatingRange
    grouped = df.groupby('RatingRange', observed=True).agg(
        total_puzzles=('correct_second_move', 'size'),
        correctly_predicted=('correct_second_move', 'sum')
    ).reset_index()

    # Calculate the percentage of correctly predicted puzzles
    grouped['percentage_correct'] = (grouped['correctly_predicted'] / grouped['total_puzzles']) * 100

    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 6))

    bar_width = 0.4
    ax1.bar(grouped['RatingRange'], grouped['total_puzzles'], width=bar_width, color='b', label='Total Puzzles')
    ax1.bar(grouped['RatingRange'], grouped['correctly_predicted'], width=bar_width, color='r', label='Correctly Predicted', alpha=0.7)

    ax1.set_xlabel('Rating Range')
    ax1.set_ylabel('Number of Puzzles')

    # Add light grid
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

    # Secondary y-axis for percentage correct
    ax2 = ax1.twinx()
    ax2.plot(grouped['RatingRange'], grouped['percentage_correct'], color='g', marker='o', label='Correctly Predicted (%)')
    ax2.set_ylabel('Correctly Predicted (%)')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.set_ylim(0, 100)

    # Adding a title and legend
    plt.title('Model 1: Correct Second Move Predictions by Rating Range')
    fig.tight_layout()

    # Combine legends from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(handles1 + handles2, labels1 + labels2, loc='upper left')

    # Improve x-axis label readability
    plt.xticks(rotation=45, ha='right')

    # Show only every 5th x-tick
    ax1.set_xticks(ax1.get_xticks()[::2])

    # Show plot
    plt.show()

plot_second_predictions(results_df)


def plot_fourth_predictions(df):

    # Define rating ranges with steps of 100
    bins = list(range(0, 2600, 100))  # From 0 to 2500 in steps of 100
    labels = [f'{i}-{i+99}' for i in range(0, 2500, 100)]
    
    # Add a column for the rating range
    df = df.copy()
    df['RatingRange'] = pd.cut(df['Rating'], bins=bins, labels=labels, right=False, include_lowest=True)
    
    # Group by RatingRange
    grouped = df.groupby('RatingRange', observed=True).agg(
        total_puzzles=('correct_fourth_move', 'size'),
        correctly_predicted=('correct_fourth_move', 'sum')
    ).reset_index()

    # Calculate the percentage of correctly predicted puzzles
    grouped['percentage_correct'] = (grouped['correctly_predicted'] / grouped['total_puzzles']) * 100

    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 6))

    bar_width = 0.4
    ax1.bar(grouped['RatingRange'], grouped['total_puzzles'], width=bar_width, color='b', label='Total Puzzles')
    ax1.bar(grouped['RatingRange'], grouped['correctly_predicted'], width=bar_width, color='r', label='Correctly Predicted', alpha=0.7)

    ax1.set_xlabel('Rating Range')
    ax1.set_ylabel('Number of Puzzles')

    # Add light grid
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

    # Secondary y-axis for percentage correct
    ax2 = ax1.twinx()
    ax2.plot(grouped['RatingRange'], grouped['percentage_correct'], color='g', marker='o', label='Correctly Predicted (%)')
    ax2.set_ylabel('Correctly Predicted (%)')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.set_ylim(0, 100)

    # Adding a title and legend
    plt.title('Model 1: Correct Fourth Move Predictions by Rating Range')
    fig.tight_layout()

    # Combine legends from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(handles1 + handles2, labels1 + labels2, loc='upper left')

    # Improve x-axis label readability
    plt.xticks(rotation=45, ha='right')

    # Show only every 5th x-tick
    ax1.set_xticks(ax1.get_xticks()[::2])

    # Show plot
    plt.show()

plot_fourth_predictions(fourth_df)



### PLOT OF ACCURACY VS ACCURACY: SECOND VS FOURTH MOVE PREDICTION ###

def plot_combined_accuracy(second_df, fourth_df):
    # Define rating ranges with steps of 100
    bins = list(range(0, 2600, 100))  # From 0 to 2500 in steps of 100
    labels = [f'{i}-{i+99}' for i in range(0, 2500, 100)]
    
    # Add a column for the rating range
    second_df = second_df.copy()
    fourth_df = fourth_df.copy()
    second_df['RatingRange'] = pd.cut(second_df['Rating'], bins=bins, labels=labels, right=False, include_lowest=True)
    fourth_df['RatingRange'] = pd.cut(fourth_df['Rating'], bins=bins, labels=labels, right=False, include_lowest=True)
    
    # Group by RatingRange
    grouped_second = second_df.groupby('RatingRange', observed=True).agg(
        total_puzzles=('correct_second_move', 'size'),
        correctly_predicted=('correct_second_move', 'sum')
    ).reset_index()
    grouped_second['percentage_correct'] = (grouped_second['correctly_predicted'] / grouped_second['total_puzzles']) * 100

    grouped_fourth = fourth_df.groupby('RatingRange', observed=True).agg(
        total_puzzles=('correct_fourth_move', 'size'),
        correctly_predicted=('correct_fourth_move', 'sum')
    ).reset_index()
    grouped_fourth['percentage_correct'] = (grouped_fourth['correctly_predicted'] / grouped_fourth['total_puzzles']) * 100

    # Calculate the Pearson correlation coefficient
    correlation, _ = pearsonr(grouped_second['percentage_correct'], grouped_fourth['percentage_correct'])
    correlation_text = f'Pearson Correlation: {correlation:.2f}'

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))

    # Add clearer grid
    ax.grid(True, which='major', linestyle='--', linewidth=1, color='gray', alpha=0.5)
    ax.grid(True, which='minor', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    ax.minorticks_on()

    # Plot percentage correct for both second and fourth moves
    ax.plot(grouped_second['RatingRange'], grouped_second['percentage_correct'], color='b', marker='o', label='Second Move Correctly Predicted (%)')
    ax.plot(grouped_fourth['RatingRange'], grouped_fourth['percentage_correct'], color='r', marker='o', label='Fourth Move Correctly Predicted (%)')
    ax.set_ylabel('Correctly Predicted (%)')
    ax.set_ylim(0, 100)

    # Adding a title and legend
    plt.title('Accuracy of Second and Fourth Move Predictions by Rating Range')
    fig.tight_layout()

    # Combine legends
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left')

    # Improve x-axis label readability
    plt.xticks(rotation=45, ha='right')

    # Show only every 5th x-tick
    ax.set_xticks(ax.get_xticks()[::2])

    # Annotate the plot with the Pearson correlation coefficient
    plt.annotate(correlation_text, xy=(0.80, 0.075), xycoords='axes fraction', fontsize=12, ha='left', va='top',
                 bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))

    # Show plot
    plt.show()

plot_combined_accuracy(second_df, fourth_df)



## PLOT OF SECOND MOVE PREDICTION ACCURACY OF MOVE THEN PIECE MODEL ###

import pandas as pd
import matplotlib.pyplot as plt

second_df.columns

def plot_second_predictions(df):
    # Create a new column to check if the predicted move is correct
    df['correct_second_move'] = df['Actual_Move'] == df['Predicted_Move']
    
    # Define rating ranges with steps of 100
    bins = list(range(0, 2600, 100))  # From 0 to 2500 in steps of 100
    labels = [f'{i}-{i+99}' for i in range(0, 2500, 100)]
    
    # Add a column for the rating range
    df = df.copy()
    df['RatingRange'] = pd.cut(df['Rating'], bins=bins, labels=labels, right=False, include_lowest=True)
    
    # Group by RatingRange
    grouped = df.groupby('RatingRange', observed=True).agg(
        total_puzzles=('correct_second_move', 'size'),
        correctly_predicted=('correct_second_move', 'sum')
    ).reset_index()

    # Calculate the percentage of correctly predicted puzzles
    grouped['percentage_correct'] = (grouped['correctly_predicted'] / grouped['total_puzzles']) * 100

    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 6))

    bar_width = 0.4
    ax1.bar(grouped['RatingRange'], grouped['total_puzzles'], width=bar_width, color='b', label='Total Puzzles')
    ax1.bar(grouped['RatingRange'], grouped['correctly_predicted'], width=bar_width, color='r', label='Correctly Predicted', alpha=0.7)

    ax1.set_xlabel('Rating Range')
    ax1.set_ylabel('Number of Puzzles')

    # Add light grid
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

    # Secondary y-axis for percentage correct
    ax2 = ax1.twinx()
    ax2.plot(grouped['RatingRange'], grouped['percentage_correct'], color='g', marker='o', label='Correctly Predicted (%)')
    ax2.set_ylabel('Correctly Predicted (%)')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.set_ylim(0, 100)

    # Adding a title and legend
    plt.title('Model 2: Correct Second Move Predictions by Rating Range')
    fig.tight_layout()

    # Combine legends from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(handles1 + handles2, labels1 + labels2, loc='upper left')

    # Improve x-axis label readability
    plt.xticks(rotation=45, ha='right')

    # Show only every 5th x-tick
    ax1.set_xticks(ax1.get_xticks()[::2])

    # Show plot
    plt.show()

plot_second_predictions(second_df)



fourth_df.columns

def plot_fourth_predictions(df):
    # Create a new column to check if the predicted fourth move is correct
    df['correct_fourth_move'] = df['Actual_Second_Move'] == df['Predicted_Second_Move']
    
    # Define rating ranges with steps of 100
    bins = list(range(0, 2600, 100))  # From 0 to 2500 in steps of 100
    labels = [f'{i}-{i+99}' for i in range(0, 2500, 100)]
    
    # Add a column for the rating range
    df = df.copy()
    df['RatingRange'] = pd.cut(df['Rating'], bins=bins, labels=labels, right=False, include_lowest=True)
    
    # Group by RatingRange
    grouped = df.groupby('RatingRange', observed=True).agg(
        total_puzzles=('correct_fourth_move', 'size'),
        correctly_predicted=('correct_fourth_move', 'sum')
    ).reset_index()

    # Calculate the percentage of correctly predicted puzzles
    grouped['percentage_correct'] = (grouped['correctly_predicted'] / grouped['total_puzzles']) * 100

    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 6))

    bar_width = 0.4
    ax1.bar(grouped['RatingRange'], grouped['total_puzzles'], width=bar_width, color='b', label='Total Puzzles')
    ax1.bar(grouped['RatingRange'], grouped['correctly_predicted'], width=bar_width, color='r', label='Correctly Predicted', alpha=0.7)

    ax1.set_xlabel('Rating Range')
    ax1.set_ylabel('Number of Puzzles')

    # Add light grid
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

    # Secondary y-axis for percentage correct
    ax2 = ax1.twinx()
    ax2.plot(grouped['RatingRange'], grouped['percentage_correct'], color='g', marker='o', label='Correctly Predicted (%)')
    ax2.set_ylabel('Correctly Predicted (%)')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.set_ylim(0, 100)

    # Adding a title and legend
    plt.title('Model 2: Correct Fourth Move Predictions by Rating Range')
    fig.tight_layout()

    # Combine legends from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(handles1 + handles2, labels1 + labels2, loc='upper left')

    # Improve x-axis label readability
    plt.xticks(rotation=45, ha='right')

    # Show only every 5th x-tick
    ax1.set_xticks(ax1.get_xticks()[::2])

    # Show plot
    plt.show()


plot_fourth_predictions(fourth_df)


fourth_df.columns




### Confusion Matrix ###
import chess
import numpy as np

# Function to get the piece from FEN
def get_piece_from_fen(fen, move):
    board = chess.Board(fen)
    uci_move = chess.Move.from_uci(move)
    piece = board.piece_at(uci_move.from_square)
    return piece.symbol() if piece else None

# Create dictionary to map PuzzleId to FEN and Moves
fen_moves_dict = df.set_index('PuzzleId')[['FEN', 'Moves']].to_dict('index')

# Add correct_piece_second_move and correct_piece_fourth_move columns
results_df['correct_piece_second_move'] = results_df.apply(
    lambda row: get_piece_from_fen(fen_moves_dict[row['PuzzleId']]['FEN'], 
                                   fen_moves_dict[row['PuzzleId']]['Moves'].split()[1]), axis=1)

results_df['correct_piece_fourth_move'] = results_df.apply(
    lambda row: get_piece_from_fen(fen_moves_dict[row['PuzzleId']]['FEN'], 
                                   fen_moves_dict[row['PuzzleId']]['Moves'].split()[3]) 
                                   if len(fen_moves_dict[row['PuzzleId']]['Moves'].split()) > 3 else None, axis=1)

# Add predicted_piece_second_move and predicted_piece_fourth_move columns
results_df['predicted_piece_second_move'] = results_df.apply(
    lambda row: get_piece_from_fen(fen_moves_dict[row['PuzzleId']]['FEN'], 
                                   row['predicted_second_move']), axis=1)

results_df['predicted_piece_fourth_move'] = results_df.apply(
    lambda row: get_piece_from_fen(fen_moves_dict[row['PuzzleId']]['FEN'], 
                                   row['predicted_fourth_move']) 
                                   if len(fen_moves_dict[row['PuzzleId']]['Moves'].split()) > 3 else None, axis=1)

# Remove rows with None values in the specified columns
results_df.dropna(subset=['correct_piece_second_move', 'correct_piece_fourth_move', 'predicted_piece_second_move', 'predicted_piece_fourth_move'], inplace=True)

results_df.columns



# Convert piece symbols to uppercase
results_df['correct_piece_second_move'] = results_df['correct_piece_second_move'].str.upper()
results_df['correct_piece_fourth_move'] = results_df['correct_piece_fourth_move'].str.upper()
results_df['predicted_piece_second_move'] = results_df['predicted_piece_second_move'].str.upper()
results_df['predicted_piece_fourth_move'] = results_df['predicted_piece_fourth_move'].str.upper()

# Unique pieces
pieces = ['R', 'N', 'P', 'Q', 'K', 'B']

# Initialize confusion matrix
combined_conf_matrix = pd.DataFrame(0, index=pieces, columns=pieces)

# Populate the confusion matrix
for _, row in results_df.iterrows():
    if row['correct_piece_second_move'] in pieces and row['predicted_piece_second_move'] in pieces:
        combined_conf_matrix.at[row['predicted_piece_second_move'], row['correct_piece_second_move']] += 1
    if row['correct_piece_fourth_move'] in pieces and row['predicted_piece_fourth_move'] in pieces:
        combined_conf_matrix.at[row['predicted_piece_fourth_move'], row['correct_piece_fourth_move']] += 1

# Display the combined matrix
print(combined_conf_matrix)


# Extract diagonal elements (correct predictions)
correct_predictions = np.diag(combined_conf_matrix)

# Calculate the total number of moves
total_moves = combined_conf_matrix.sum().sum()

# Calculate accuracy
accuracy = correct_predictions.sum() / total_moves * 100

print(f"Accuracy of correctly predicted moves: {accuracy:.2f}%")


import numpy as np

results_df.columns

# Remove '@' and everything after it in Actual_Piece and Predicted_Piece columns
results_df['Actual_Piece_Second_Move'] = results_df['Actual_Piece_Second_Move'].str.split('@').str[0].str.upper()
results_df['Predicted_Piece_Second_Move'] = results_df['Predicted_Piece_Second_Move'].str.split('@').str[0].str.upper()

# Create confusion matrix
conf_matrix = pd.crosstab(results_df['Predicted_Piece_Second_Move'], results_df['Actual_Piece_Second_Move'], rownames=['Predicted'], colnames=['Actual'])

# Define all possible pieces
all_pieces = ['R', 'N', 'P', 'Q', 'K', 'B']

# Reindex the confusion matrix to include all pieces
conf_matrix = conf_matrix.reindex(index=all_pieces, columns=all_pieces, fill_value=0)

conf_matrix

# Calculate the number of correct predictions
correct_predictions = np.diag(conf_matrix).sum()

# Calculate the total number of predictions
total_predictions = conf_matrix.sum().sum()

# Calculate the accuracy
accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

print(f"Accuracy: {accuracy * 100:.2f}%")