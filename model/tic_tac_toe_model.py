!pip install numpy torch

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def transform_grid(actions, move_count):
    state = [None] * 9
    for action in actions:
        if action['moveStep'] < move_count:
            square = action['position']
            state[square] = action['token']

    transformed = []
    for square in state:
        if square == 'X':
            transformed.append([1, 0, 0])
        elif square == 'O':
            transformed.append([0, 1, 0])
        else:
            transformed.append([0, 0, 1])

    matrix = np.array(transformed).reshape(3, 3, 3)
    matrix = np.transpose(matrix, (2, 0, 1))
    return matrix

class GridGameDataset(Dataset):
    def __init__(self, match_data):
        self.records = []
        for session in match_data:
            actions = session['moves']
            actions = sorted(actions, key=lambda act: act['moveNumber'])
            for action in actions:
                turn = action['moveNumber']
                grid_snapshot = transform_grid(actions, turn)
                target = action['cell']
                self.records.append((grid_snapshot, target))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        grid, target = self.records[index]
        grid_tensor = torch.tensor(grid, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.long)
        return grid_tensor, target_tensor

game_dataset = GridGameDataset(data)
game_dataloader = DataLoader(game_dataset, batch_size=32, shuffle=True)

print(f"Total training samples: {len(game_dataset)}")

class TicTacToeModel(nn.Module):
    def __init__(self):
        super(TicTacToeModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, padding=1)

        self.embedding_dim = 64

        self.flatten = nn.Flatten(2)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=4, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.fc = nn.Linear(25 * self.embedding_dim, 9)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)

        x = x.permute(2, 0, 1)
        x = self.transformer_encoder(x)

        x = x.permute(1, 0, 2).reshape(x.size(1), -1)
        out = self.fc(x)
        return out

model = TicTacToeModel()

# Set up device, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion_function = nn.CrossEntropyLoss()
optim_function = optim.Adam(model.parameters(), lr=0.001)

training_epochs = 10

for iteration in range(training_epochs):
    model.train()
    cumulative_loss = 0.0
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optim_function.zero_grad()
        predictions = model(inputs)
        error = criterion_function(predictions, targets)
        error.backward()
        optim_function.step()

        cumulative_loss += error.item() * inputs.size(0)

    average_loss = cumulative_loss / len(dataset)
    print(f'Epoch {iteration+1}/{training_epochs}, Loss: {average_loss:.4f}')

model.eval()
sample_board, sample_label = dataset[0]
with torch.no_grad():
    sample_board = sample_board.unsqueeze(0).to(device)
    output = model(sample_board)
    predicted_move = torch.argmax(output, dim=1).item()

print("True move:", sample_label)
print("Predicted move:", predicted_move)

# Save the model's state dictionary
torch.save(model.state_dict(), 'tic_tac_toe_model.pt')
print("Model saved as tic_tac_toe_model.pt")

from google.colab import files
files.download("tic_tac_toe_model.pt")

loaded_model = TicTacToeModel()

loaded_model.load_state_dict(torch.load('tic_tac_toe_model.pt', map_location=device))
loaded_model.to(device)
loaded_model.eval()

print("Model loaded and ready for inference.")

# Retrieve a sample board state from the dataset
sample_board, sample_label = dataset[0]

sample_board = sample_board.unsqueeze(0).to(device)

# Perform inference
with torch.no_grad():
    output = loaded_model(sample_board)
    predicted_move = torch.argmax(output, dim=1).item()

print("True move (cell index):", sample_label)
print("Predicted move (cell index):", predicted_move)

custom_moves = [
    {"moveNumber": 1, "cell": 0, "symbol": "X"},
    {"moveNumber": 2, "cell": 4, "symbol": "O"},
    {"moveNumber": 3, "cell": 1, "symbol": "X"},
    {"moveNumber": 4, "cell": 8, "symbol": "O"},
    {"moveNumber": 5, "cell": 6, "symbol": "X"},

]
custom_board = transform_grid(custom_moves, current_move_num=6)
custom_board_tensor = torch.tensor(custom_board, dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    custom_output = loaded_model(custom_board_tensor)
    custom_predicted_move = torch.argmax(custom_output, dim=1).item()

print("For the custom board state, predicted move (cell index):", custom_predicted_move)

import numpy as np
import torch
import torch.nn.functional as F

def check_winner(board):
    winning_combinations = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8],
        [0, 4, 8],
        [2, 4, 6]
    ]
    for combo in winning_combinations:
        a, b, c = combo
        if board[a] is not None and board[a] == board[b] == board[c]:
            return board[a]
    return None

def simulate_game_random(model, starting_symbol='O'):
    board = [None] * 9
    current_symbol = starting_symbol
    moves = []

    for move_num in range(1, 10):
        winner = check_winner(board)
        if winner is not None:
            print(f"Game ended: {winner} wins!")
            break

        if None not in board:
            print("Game ended in a draw!")
            break

        fake_moves = [
            {"moveNumber": i+1, "cell": m['cell'], "symbol": m['symbol']}
            for i, m in enumerate(moves)
        ]
        board_input = transform_grid(fake_moves, move_num)
        board_tensor = torch.tensor(board_input, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(board_tensor)
            probs = F.softmax(output, dim=1).cpu().numpy().flatten()
        valid_cells = [i for i, cell in enumerate(board) if cell is None]
        if len(valid_cells) == 0:
            break

        valid_probs = np.array([probs[i] for i in valid_cells])
        valid_probs = valid_probs / valid_probs.sum()

        chosen_cell = np.random.choice(valid_cells, p=valid_probs)
        board[chosen_cell] = current_symbol
        moves.append({"moveNumber": move_num, "cell": chosen_cell, "symbol": current_symbol})

        print(f"Move {move_num}: Player {current_symbol} plays cell {chosen_cell}")
        print(np.array(board).reshape(3, 3))

        winner = check_winner(board)
        if winner is not None:
            print(f"Game ended: {winner} wins!")
            break
        if None not in board:
            print("Game ended in a draw!")
            break

        current_symbol = 'O' if current_symbol == 'X' else 'X'

    print("Game complete.")
    return moves

simulate_game_random(loaded_model)

import numpy as np
import torch
import torch.nn.functional as F

def check_winner(board):
    winning_combinations = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8],
        [0, 4, 8],
        [2, 4, 6]
    ]
    for combo in winning_combinations:
        a, b, c = combo
        if board[a] is not None and board[a] == board[b] == board[c]:
            return board[a]
    return None


def get_winning_moves(board, player):
    winning_moves = []
    for cell in range(9):
        if board[cell] is None:
            temp_board = board.copy()
            temp_board[cell] = player
            if check_winner(temp_board) == player:
                winning_moves.append(cell)
    return winning_moves

def simulate_game_random(model, starting_symbol='O'):
    board = [None] * 9
    current_symbol = starting_symbol
    moves = []

    total_winning_situations = 0
    correct_winning_moves = 0

    for move_num in range(1, 10):

        winner = check_winner(board)
        if winner is not None:
            print(f"Game ended: {winner} wins!")
            break
        if None not in board:
            print("Game ended in a draw!")
            break
        winning_moves = get_winning_moves(board, current_symbol)
        if winning_moves:
            total_winning_situations += 1

        fake_moves = [
            {"moveNumber": i+1, "cell": m['cell'], "symbol": m['symbol']}
            for i, m in enumerate(moves)
        ]
        board_input = transform_grid(fake_moves, move_num)
        board_tensor = torch.tensor(board_input, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(board_tensor)
            probs = F.softmax(output, dim=1).cpu().numpy().flatten()

        valid_cells = [i for i, cell in enumerate(board) if cell is None]
        if len(valid_cells) == 0:
            break
        valid_probs = np.array([probs[i] for i in valid_cells])
        valid_probs = valid_probs / valid_probs.sum()

        chosen_cell = np.random.choice(valid_cells, p=valid_probs)

        if winning_moves:
            if chosen_cell in winning_moves:
                correct_winning_moves += 1
            else:
                print(f"Winning move available {winning_moves}, but model chose {chosen_cell}.")

        board[chosen_cell] = current_symbol
        moves.append({"moveNumber": move_num, "cell": chosen_cell, "symbol": current_symbol})

        print(f"Move {move_num}: Player {current_symbol} plays cell {chosen_cell}")
        print(np.array(board).reshape(3, 3))

        winner = check_winner(board)
        if winner is not None:
            print(f"Game ended: {winner} wins!")
            break
        if None not in board:
            print("Game ended in a draw!")
            break
        current_symbol = 'O' if current_symbol == 'X' else 'X'

    print("Game complete.")

    if total_winning_situations > 0:
        accuracy = correct_winning_moves / total_winning_situations * 100
        print(f"Winning Move Accuracy: {accuracy:.2f}% ({correct_winning_moves} / {total_winning_situations} times)")
    else:
        print("No winning move situations encountered.")

    return moves
simulate_game_random(loaded_model)

def simulate_game_random_with_accuracy(model, starting_symbol='O'):
    board = [None] * 9
    current_symbol = starting_symbol
    moves = []

    total_winning_situations = 0
    correct_winning_moves = 0

    for move_num in range(1, 10):
        if check_winner(board) is not None or None not in board:
            break
        winning_moves = get_winning_moves(board, current_symbol)
        if winning_moves:
            total_winning_situations += 1
        fake_moves = [
            {"moveNumber": i + 1, "cell": m['cell'], "symbol": m['symbol']}
            for i, m in enumerate(moves)
        ]
        board_input = transform_grid(fake_moves, move_num)
        board_tensor = torch.tensor(board_input, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(board_tensor)
            probs = F.softmax(output, dim=1).cpu().numpy().flatten()

        valid_cells = [i for i, cell in enumerate(board) if cell is None]
        if not valid_cells:
            break
        valid_probs = np.array([probs[i] for i in valid_cells])
        valid_probs = valid_probs / valid_probs.sum()

        chosen_cell = np.random.choice(valid_cells, p=valid_probs)
        if winning_moves and chosen_cell in winning_moves:
            correct_winning_moves += 1
        board[chosen_cell] = current_symbol
        moves.append({"moveNumber": move_num, "cell": chosen_cell, "symbol": current_symbol})
        if check_winner(board) is not None:
            break
        current_symbol = 'O' if current_symbol == 'X' else 'X'
    return moves, total_winning_situations, correct_winning_moves

def test_simulation_games(model, num_games=50, starting_symbol='O'):
    total_ws = 0
    total_correct = 0
    for i in range(num_games):
        _, ws, correct = simulate_game_random_with_accuracy(model, starting_symbol)
        total_ws += ws
        total_correct += correct

    if total_ws > 0:
        avg_accuracy = total_correct / total_ws * 100
    else:
        avg_accuracy = 0.0
    print(f"Average Winning Move Accuracy over {num_games} games: {avg_accuracy:.2f}%")

test_simulation_games(loaded_model)