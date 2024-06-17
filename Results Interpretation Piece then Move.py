import pandas as pd
import matplotlib.pyplot as plt


### FIRST PIECE ACCURACY ###

def get_results_df():
    results_df = pd.read_csv('ONLY SECOND PIECE AND MOVE.csv')
    return results_df

df = get_results_df()
df


def plot_piece_predictions(df):
    # Create a new column to check if the predicted piece is correct
    df['correct_piece_move'] = df['Actual_Second_Piece'] == df['Predicted_Second_Piece']
    
    # Define rating ranges with steps of 100
    bins = list(range(0, 2600, 100))  # From 0 to 2500 in steps of 100
    labels = [f'{i}-{i+99}' for i in range(0, 2500, 100)]
    
    # Add a column for the rating range
    df = df.copy()
    df['RatingRange'] = pd.cut(df['Rating'], bins=bins, labels=labels, right=False, include_lowest=True)
    
    # Group by RatingRange
    grouped = df.groupby('RatingRange', observed=True).agg(
        total_puzzles=('correct_piece_move', 'size'),
        correctly_predicted=('correct_piece_move', 'sum')
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
    plt.title('Model 2: Correct Second Piece Predictions by Rating Range')
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


plot_piece_predictions(df)

df[df['Actual_Piece'].str.contains('P')]



# Extract the piece from the piece_with_position string
df['Actual_Piece_Type'] = df['Actual_Piece'].str[0].str.lower()
df['Predicted_Piece_Type'] = df['Predicted_Piece'].str[0].str.lower()

# Initialize a dictionary to hold counts for correct and incorrect predictions
piece_counts = {piece: {'correct': 0, 'incorrect': 0} for piece in ['r', 'q', 'n', 'b', 'p', 'k']}

# Populate the dictionary with counts
for index, row in df.iterrows():
    actual_piece = row['Actual_Piece_Type']
    predicted_piece = row['Predicted_Piece_Type']
    
    if actual_piece == predicted_piece:
        piece_counts[actual_piece]['correct'] += 1
    else:
        piece_counts[actual_piece]['incorrect'] += 1

# Convert the dictionary to a DataFrame
piece_counts_df = pd.DataFrame.from_dict(piece_counts, orient='index')
piece_counts_df = piece_counts_df.reset_index().rename(columns={'index': 'Piece', 'correct': 'Correct Predictions', 'incorrect': 'Incorrect Predictions'})

print(piece_counts_df)
df.head(50)