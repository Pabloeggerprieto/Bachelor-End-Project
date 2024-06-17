import numpy as np
import pandas as pd
import chess
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

from Data.LoadingData import DataLoader

# Initialize DataLoader with the file path
loader = DataLoader('VERSION 1/Data/lichess_db_puzzle.csv.zst')

# Load data
df = loader.backrank_and_m2()
df = df.head(1000)

# Update board with a move
def update_board(fen, move):
    if move is None:
        return fen
    board = chess.Board(fen)
    move = chess.Move.from_uci(move)
    if move in board.legal_moves:
        board.push(move)
    return board.fen()

# Parse FEN into a numeric matrix for CNN input
def parse_fen(fen):
    pieces = {'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6, 'P': -1, 'N': -2, 'B': -3, 'R': -4, 'Q': -5, 'K': -6, '.': 0}
    board = np.zeros((8, 8), dtype=int)
    rows = fen.split()[0].split('/')
    for i, row in enumerate(rows):
        col = 0
        for char in row:
            if char.isdigit():
                col += int(char)  # Move over empty squares
            else:
                board[i, col] = pieces[char]
                col += 1
    return board

# Extract the piece to move based on the move
def get_piece_to_move(fen, move):
    if move is None:
        return None
    board = chess.Board(fen)
    move = chess.Move.from_uci(move)
    piece = board.piece_at(move.from_square)
    return piece.symbol() if piece else None

# Extract the piece and its position
def get_piece_and_position(fen, move):
    if move is None:
        return None
    board = chess.Board(fen)
    move = chess.Move.from_uci(move)
    piece = board.piece_at(move.from_square)
    return f"{piece.symbol()}@{chess.square_name(move.from_square)}" if piece else None

# Check if a move results in a checkmate
def is_checkmate(fen, move):
    if move is None:
        return False
    board = chess.Board(fen)
    move = chess.Move.from_uci(move)
    if move in board.legal_moves:
        board.push(move)
        return board.is_checkmate()
    return False

# Prepare data for second piece and move prediction
def prepare_data_for_second_prediction(df):
    df['FEN_after_first_move'] = df.apply(lambda row: update_board(row['FEN'], row['Moves'].split()[0]), axis=1)
    df['FEN_after_second_move'] = df.apply(lambda row: update_board(row['FEN_after_first_move'], row['Moves'].split()[1]), axis=1)
    df['FEN_after_opponent_second_move'] = df.apply(lambda row: update_board(row['FEN_after_second_move'], row['Moves'].split()[2]), axis=1)
    df['piece_to_move2'] = df.apply(lambda row: get_piece_and_position(row['FEN_after_opponent_second_move'], row['Moves'].split()[3]) if len(row['Moves'].split()) > 3 else None, axis=1)
    df['board_matrix2'] = df['FEN_after_opponent_second_move'].apply(parse_fen)
    df['is_checkmate'] = df.apply(lambda row: is_checkmate(row['FEN_after_opponent_second_move'], row['Moves'].split()[3]) if len(row['Moves'].split()) > 3 else False, axis=1)
    return df

df = prepare_data_for_second_prediction(df)

# Collecting all possible moves made by each piece from the legal moves for second moves
def collect_all_legal_moves(df):
    legal_moves = set()
    for index, row in df.iterrows():
        # Second move
        board = chess.Board(row['FEN_after_opponent_second_move'])
        moves = [move.uci() for move in board.legal_moves if board.piece_at(move.from_square) and board.piece_at(move.from_square).symbol() == row['piece_to_move2'].split('@')[0]]
        legal_moves.update(moves)
        
    return list(legal_moves)

# Ensure all possible moves are included for encoding
all_legal_moves = collect_all_legal_moves(df)
label_encoder_moves = LabelEncoder()
label_encoder_moves.fit(all_legal_moves)

# Limit the second piece prediction to rooks, queens, and promoted pawns
limited_pieces = ['r', 'q', 'R', 'Q', 'P']
df['piece_to_move2_limited'] = df['piece_to_move2'].apply(lambda x: x if x and x.split('@')[0] in limited_pieces else None)

# Drop rows with None values in 'piece_to_move2_limited' to avoid issues with label encoding
df = df.dropna(subset=['piece_to_move2_limited'])

# Remove classes with very few samples
min_samples = 2
df = df.groupby('piece_to_move2_limited').filter(lambda x: len(x) >= min_samples)

# Label encoding pieces and split data for the piece prediction model
label_encoder_pieces2 = LabelEncoder()
df['encoded_piece2'] = label_encoder_pieces2.fit_transform(df['piece_to_move2_limited'])
X2 = np.array(df['board_matrix2'].tolist()).reshape((-1, 8, 8, 1))
y2 = df['encoded_piece2']

# Store the original indices
original_indices = np.arange(len(df))

# Perform the train-test split and keep track of the indices
X_train_piece2, X_test_piece2, y_train_piece2, y_test_piece2, train_indices_piece2, test_indices_piece2 = train_test_split(
    X2, y2, original_indices, test_size=0.2, random_state=42, stratify=y2
)

# Model to predict the second piece
model_piece2 = Sequential([
    Input(shape=(8, 8, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2), padding='same'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder_pieces2.classes_), activation='softmax')
])
model_piece2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_piece2.fit(X_train_piece2, y_train_piece2, epochs=10, batch_size=16, validation_split=0.2)

# Function to filter moves for a given piece
def filter_moves_for_piece(fen, piece_with_position):
    if piece_with_position is None:
        return []
    piece, position = piece_with_position.split('@')
    square = chess.parse_square(position)
    board = chess.Board(fen)
    return [move.uci() for move in board.legal_moves if move.from_square == square and board.piece_at(move.from_square).symbol() == piece]

# Prepare data for the second move model
def prepare_move_data(df):
    X2 = []
    y2 = []
    checkmate_labels = []
    for index, row in df.iterrows():
        fen = row['FEN_after_opponent_second_move']
        piece_with_position = row['piece_to_move2_limited']
        target_move = row['Moves'].split()[3] if len(row['Moves'].split()) > 3 else None

        filtered_moves = filter_moves_for_piece(fen, piece_with_position)
        if target_move in filtered_moves:
            board_matrix = parse_fen(fen)
            X2.append(board_matrix)
            y2.append(label_encoder_moves.transform([target_move])[0])
            checkmate_labels.append(row['is_checkmate'])

    X2 = np.array(X2).reshape((-1, 8, 8, 1))
    y2 = to_categorical(y2, num_classes=len(label_encoder_moves.classes_))
    checkmate_labels = np.array(checkmate_labels).astype(int)
    return X2, y2, checkmate_labels

X_move2, y_move2, checkmate_labels = prepare_move_data(df)
original_indices_move2 = np.arange(len(X_move2))

# Perform the train-test split and keep track of the indices without stratification
X_train_move2, X_test_move2, y_train_move2, y_test_move2, train_indices_move2, test_indices_move2, train_checkmate_labels, test_checkmate_labels = train_test_split(
    X_move2, y_move2, original_indices_move2, checkmate_labels, test_size=0.2, random_state=42
)

# Model to predict the second move
model_move2 = Sequential([
    Input(shape=(8, 8, 1)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2), padding='same'),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder_moves.classes_), activation='softmax')
])

model_move2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_move2.fit(X_train_move2, y_train_move2, epochs=10, batch_size=16, validation_split=0.2)


def predict_piece_then_move(fen, target_piece_with_position, target_move, model_piece, model_move, label_encoder_pieces, label_encoder_moves):
    # Parse FEN into board matrix
    board_matrix = parse_fen(fen).reshape(1, 8, 8, 1)
    
    # Predict the piece to move
    piece_pred = model_piece.predict(board_matrix)
    predicted_piece_idx = np.argmax(piece_pred)
    
    if predicted_piece_idx < len(label_encoder_pieces.classes_):
        predicted_piece_with_position = label_encoder_pieces.inverse_transform([predicted_piece_idx])[0]
    else:
        return None, None

    if predicted_piece_with_position == target_piece_with_position:
        # Get legal moves for the predicted piece
        piece, position = predicted_piece_with_position.split('@')
        square = chess.parse_square(position)
        board = chess.Board(fen)
        legal_moves = [move for move in board.legal_moves 
                       if move.from_square == square and board.piece_at(move.from_square).symbol() == piece]

        # Debugging information: Check if the target move is in the list of legal moves
        print(f"Target move: {target_move}, Legal moves: {[move.uci() for move in legal_moves]}")

        if target_move not in [move.uci() for move in legal_moves]:
            print(f"Warning: Target move {target_move} not in legal moves")

        # Check for checkmate moves
        checkmate_moves = []
        for move in legal_moves:
            board.push(move)
            if board.is_checkmate():
                checkmate_moves.append(move)
            board.pop()

        # If there are any checkmate moves, prioritize them
        if checkmate_moves:
            checkmate_move = checkmate_moves[0].uci()
            print(f"Checkmate move found: {checkmate_move}")
            return predicted_piece_with_position, checkmate_move

        if legal_moves:
            # Encode legal moves using the label encoder
            legal_moves_uci = [move.uci() for move in legal_moves]
            encoded_legal_moves = label_encoder_moves.transform(legal_moves_uci)
            
            # Create a dummy array to predict the legal moves
            dummy_input = np.zeros((len(encoded_legal_moves), 8, 8, 1))
            
            # Predict the move using the filtered legal moves only if the predicted piece is correct
            move_preds = model_move.predict(dummy_input)
            predicted_move_idx = np.argmax(move_preds)

            # Ensure the predicted move index is within the range of legal moves
            predicted_move_idx = min(predicted_move_idx, len(encoded_legal_moves) - 1)

            # Decode the predicted move index
            predicted_move = legal_moves_uci[predicted_move_idx]
            return predicted_piece_with_position, predicted_move

    return predicted_piece_with_position, None

# Evaluate predictions for second piece and move with debugging information
def evaluate_predictions(df, model_piece, model_move, label_encoder_pieces, label_encoder_moves, piece_test_indices):
    correct_second_piece_predictions = 0
    correct_second_move_predictions = 0
    total_predictions = 0
    move_predictions_made = 0

    # List to accumulate results
    results = []

    # Verify indices are within bounds
    valid_indices = [index for index in piece_test_indices if 0 <= index < len(df)]
    if len(valid_indices) != len(piece_test_indices):
        print(f"Some indices were out of bounds: {set(piece_test_indices) - set(valid_indices)}")

    for index in tqdm(valid_indices, desc="Evaluating piece predictions"):
        row = df.iloc[index]
        initial_fen = row['FEN']
        target_moves = row['Moves'].split()

        # Apply opponent's first move
        fen_after_opponent_first_move = update_board(initial_fen, target_moves[0])

        # Apply first move
        fen_after_first_move = update_board(fen_after_opponent_first_move, target_moves[1] if len(target_moves) > 1 else None)

        # Apply opponent's second move
        fen_after_opponent_second_move = update_board(fen_after_first_move, target_moves[2] if len(target_moves) > 2 else None)

        # Second move predictions (fourth move in the sequence)
        fen_after_third_move = fen_after_opponent_second_move
        target_second_piece_with_position = row['piece_to_move2_limited']
        target_second_move = target_moves[3] if len(target_moves) > 3 else None

        predicted_second_piece_with_position, predicted_second_move = predict_piece_then_move(fen_after_third_move, target_second_piece_with_position, target_second_move, model_piece, model_move, label_encoder_pieces, label_encoder_moves)

        if predicted_second_piece_with_position == target_second_piece_with_position:
            correct_second_piece_predictions += 1
            move_predictions_made += 1  # Only count move predictions if piece is correct

            if predicted_second_move == target_second_move:
                correct_second_move_predictions += 1

        # Accumulate results
        results.append({
            'PuzzleId': row['PuzzleId'],
            'GameUrl': row['GameUrl'],
            'Popularity': row['Popularity'],
            'Rating': row['Rating'],
            'Actual_Second_Piece': target_second_piece_with_position,
            'Predicted_Second_Piece': predicted_second_piece_with_position,
            'Actual_Second_Move': target_second_move,
            'Predicted_Second_Move': predicted_second_move
        })

        total_predictions += 1

    second_piece_accuracy = correct_second_piece_predictions / total_predictions if total_predictions > 0 else 0
    second_move_accuracy = correct_second_move_predictions / move_predictions_made if move_predictions_made > 0 else 0

    # Debug: Print the number of predictions made and correct predictions
    print(f"Total predictions: {total_predictions}")
    print(f"Move predictions made: {move_predictions_made}")
    print(f"Correct second piece predictions: {correct_second_piece_predictions}")
    print(f"Correct second move predictions: {correct_second_move_predictions}")

    # Create DataFrame from results list
    results_piece_move_df = pd.DataFrame(results)

    return second_piece_accuracy, second_move_accuracy, results_piece_move_df

# Running evaluation on the test set with debugging information
piece_test_indices = test_indices_piece2.tolist()
second_piece_accuracy, second_move_accuracy, results_piece_move_df = evaluate_predictions(
    df, model_piece2, model_move2, label_encoder_pieces2, label_encoder_moves, piece_test_indices
)

print(f"Second piece prediction accuracy: {second_piece_accuracy * 100:.2f}%")
print(f"Second move prediction accuracy: {second_move_accuracy * 100:.2f}%")
