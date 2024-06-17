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

# Extract the piece and its position
def get_piece_and_position(fen, move):
    if move is None:
        return None
    board = chess.Board(fen)
    move = chess.Move.from_uci(move)
    piece = board.piece_at(move.from_square)
    return f"{piece.symbol()}@{chess.square_name(move.from_square)}" if piece else None

# Function to check if a move results in a capture
def is_capture(fen, move):
    if move is None:
        return False
    board = chess.Board(fen)
    move = chess.Move.from_uci(move)
    if board.is_capture(move):
        return True
    return False

# Prepare data for piece prediction
def prepare_data_for_piece_prediction(df):
    df['FEN_after_first_move'] = df.apply(lambda row: update_board(row['FEN'], row['Moves'].split()[0]), axis=1)
    df['piece_to_move'] = df.apply(lambda row: get_piece_and_position(row['FEN_after_first_move'], row['Moves'].split()[1]) if len(row['Moves'].split()) > 1 else None, axis=1)
    df['board_matrix'] = df['FEN_after_first_move'].apply(parse_fen)
    df['is_capture'] = df.apply(lambda row: is_capture(row['FEN_after_first_move'], row['Moves'].split()[1]) if len(row['Moves'].split()) > 1 else False, axis=1)
    return df

df = prepare_data_for_piece_prediction(df)

# Collecting all possible moves made by each piece from the legal moves for first moves
def collect_all_legal_moves(df):
    legal_moves = set()
    for index, row in df.iterrows():
        # First move
        board = chess.Board(row['FEN_after_first_move'])
        moves = [move.uci() for move in board.legal_moves if board.piece_at(move.from_square) and board.piece_at(move.from_square).symbol() == row['piece_to_move'].split('@')[0]]
        legal_moves.update(moves)
        
    return list(legal_moves)

# Ensure all possible moves are included for encoding
all_legal_moves = collect_all_legal_moves(df)
label_encoder_moves = LabelEncoder()
label_encoder_moves.fit(all_legal_moves)

# Limit the piece prediction to rooks, queens, and promoted pawns
limited_pieces = ['r', 'q', 'R', 'Q', 'P']
df['piece_to_move_limited'] = df['piece_to_move'].apply(lambda x: x if x and x.split('@')[0] in limited_pieces else None)

# Drop rows with None values in 'piece_to_move_limited' to avoid issues with label encoding
df = df.dropna(subset=['piece_to_move_limited'])

# Remove classes with very few samples
min_samples = 2
df = df.groupby('piece_to_move_limited').filter(lambda x: len(x) >= min_samples)

# Label encoding pieces and split data for the piece prediction model
label_encoder_pieces = LabelEncoder()
df['encoded_piece'] = label_encoder_pieces.fit_transform(df['piece_to_move_limited'])
X = np.array(df['board_matrix'].tolist()).reshape((-1, 8, 8, 1))
y = df['encoded_piece']

# Store the original indices
original_indices = np.arange(len(df))

# Perform the train-test split and keep track of the indices
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
    X, y, original_indices, test_size=0.2, random_state=42, stratify=y
)

# Model to predict the piece
model_piece = Sequential([
    Input(shape=(8, 8, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2), padding='same'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder_pieces.classes_), activation='softmax')
])
model_piece.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_piece.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.2)

# Function to filter moves for a given piece
def filter_moves_for_piece(fen, piece_with_position):
    if piece_with_position is None:
        return []
    piece, position = piece_with_position.split('@')
    square = chess.parse_square(position)
    board = chess.Board(fen)
    return [move.uci() for move in board.legal_moves if move.from_square == square and board.piece_at(move.from_square).symbol() == piece]

# Function to parse FEN and include capturing information
def parse_fen_with_capture(fen, is_capture):
    board = parse_fen(fen)
    capture_channel = np.full(board.shape, int(is_capture), dtype=int)  # Add a channel with capturing information
    return np.stack([board, capture_channel], axis=-1)

# Prepare data for the move model with capturing information
def prepare_move_data_with_capture(df):
    X = []
    y = []
    original_indices_move = []
    for index, row in df.iterrows():
        fen = row['FEN_after_first_move']
        piece_with_position = row['piece_to_move_limited']
        target_move = row['Moves'].split()[1] if len(row['Moves'].split()) > 1 else None
        is_capture = row['is_capture']

        filtered_moves = filter_moves_for_piece(fen, piece_with_position)
        if target_move in filtered_moves:
            board_matrix_with_capture = parse_fen_with_capture(fen, is_capture)
            X.append(board_matrix_with_capture)
            y.append(label_encoder_moves.transform([target_move])[0])
            original_indices_move.append(index)

    X = np.array(X)
    y = to_categorical(y, num_classes=len(label_encoder_moves.classes_))
    return X, y, np.array(original_indices_move)

X_move, y_move, original_indices_move = prepare_move_data_with_capture(df)

# Perform the train-test split and keep track of the indices without stratification
X_train_move, X_test_move, y_train_move, y_test_move, train_indices_move, test_indices_move = train_test_split(
    X_move, y_move, original_indices_move, test_size=0.2, random_state=42
)

# Model to predict the move
model_move = Sequential([
    Input(shape=(8, 8, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2), padding='same'),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder_moves.classes_), activation='softmax')
])

model_move.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_move.fit(X_train_move, y_train_move, epochs=10, batch_size=16, validation_split=0.2)

# Prediction function for piece and then move
def predict_piece_then_move(fen, target_piece_with_position, is_capture, model_piece, model_move, label_encoder_pieces, label_encoder_moves):
    # Parse FEN into board matrix with capture information
    board_matrix_with_capture = parse_fen_with_capture(fen, is_capture).reshape(1, 8, 8, 2)
    
    # Predict the piece to move
    piece_pred = model_piece.predict(board_matrix_with_capture[:, :, :, :1])
    predicted_piece_idx = np.argmax(piece_pred)
    predicted_piece_with_position = label_encoder_pieces.inverse_transform([predicted_piece_idx])[0]

    if predicted_piece_with_position == target_piece_with_position:
        # Get legal moves for the predicted piece
        piece, position = predicted_piece_with_position.split('@')
        square = chess.parse_square(position)
        board = chess.Board(fen)
        legal_moves = [move for move in board.legal_moves 
                       if move.from_square == square and board.piece_at(move.from_square).symbol() == piece]

        # Check for capturing moves and prioritize them
        capturing_moves = [move for move in legal_moves if board.is_capture(move)]
        if capturing_moves:
            return predicted_piece_with_position, capturing_moves[0].uci()

        if legal_moves:
            # Encode legal moves using the label encoder
            legal_moves_uci = [move.uci() for move in legal_moves]
            encoded_legal_moves = label_encoder_moves.transform(legal_moves_uci)
            
            # Create a dummy array to predict the legal moves
            dummy_input = np.zeros((len(encoded_legal_moves), 8, 8, 2))
            
            # Predict the move using the filtered legal moves only if the predicted piece is correct
            move_preds = model_move.predict(dummy_input)
            predicted_move_idx = np.argmax(move_preds)

            # Ensure the predicted move index is within the range of legal moves
            predicted_move_idx = min(predicted_move_idx, len(legal_moves) - 1)

            # Decode the predicted move index
            predicted_move = legal_moves_uci[predicted_move_idx]
            return predicted_piece_with_position, predicted_move

    return predicted_piece_with_position, None

# Evaluate predictions for piece and move
def evaluate_predictions(df, model_piece, model_move, label_encoder_pieces, label_encoder_moves, piece_test_indices):
    correct_piece_predictions = 0
    correct_move_predictions = 0
    total_predictions = 0
    move_predictions_made = 0

    # List to accumulate results
    results = []

    for index in tqdm(piece_test_indices, desc="Evaluating piece predictions"):
        row = df.iloc[index]
        initial_fen = row['FEN']
        target_moves = row['Moves'].split()
        
        # Apply opponent's first move
        fen_after_opponent_first_move = update_board(initial_fen, target_moves[0])

        # First move predictions (second move in the sequence)
        fen_after_first_move = fen_after_opponent_first_move
        target_piece_with_position = row['piece_to_move']
        target_move = target_moves[1] if len(target_moves) > 1 else None
        is_capture = row['is_capture']

        predicted_piece_with_position, predicted_move = predict_piece_then_move(fen_after_first_move, target_piece_with_position, is_capture, model_piece, model_move, label_encoder_pieces, label_encoder_moves)

        if predicted_piece_with_position == target_piece_with_position:
            correct_piece_predictions += 1
            move_predictions_made += 1  # Only count move predictions if piece is correct

            if predicted_move == target_move:
                correct_move_predictions += 1

        # Accumulate results
        results.append({
            'PuzzleId': row['PuzzleId'],
            'GameUrl': row['GameUrl'],
            'Popularity': row['Popularity'],
            'Rating': row['Rating'],
            'Actual_Piece': target_piece_with_position,
            'Predicted_Piece': predicted_piece_with_position,
            'Actual_Move': target_move,
            'Predicted_Move': predicted_move
        })

        total_predictions += 1

    piece_accuracy = correct_piece_predictions / total_predictions
    move_accuracy = correct_move_predictions / move_predictions_made if move_predictions_made > 0 else 0

    # Create DataFrame from results list
    results_piece_move_df = pd.DataFrame(results)

    return piece_accuracy, move_accuracy, results_piece_move_df

# Running evaluation on the test set
piece_test_indices = test_indices.tolist()
piece_accuracy, move_accuracy, results_piece_move_df = evaluate_predictions(
    df, model_piece, model_move, label_encoder_pieces, label_encoder_moves, piece_test_indices
)

print(f"Piece prediction accuracy: {piece_accuracy * 100:.2f}%")
print(f"Move prediction accuracy: {move_accuracy * 100:.2f}%")
