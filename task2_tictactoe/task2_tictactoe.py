import math

def print_board(board):
    print("\n")
    for i in range(0, 9, 3):
        print(f" {board[i]} | {board[i+1]} | {board[i+2]} ")
        if i < 6:
            print("-----------")
    print("\n")

def check_winner(board, player):
    win_conditions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8], # rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8], # cols
        [0, 4, 8], [2, 4, 6]             # diagonals
    ]
    return any(all(board[i] == player for i in combo) for combo in win_conditions)

def is_draw(board):
    return ' ' not in board

def get_available_moves(board):
    return [i for i, spot in enumerate(board) if spot == ' ']

def minimax(board, depth, is_maximizing, alpha, beta):
    # Base cases for terminal states
    if check_winner(board, 'X'): return 10 - depth
    if check_winner(board, 'O'): return -10 + depth
    if is_draw(board): return 0

    if is_maximizing:
        best_score = -math.inf
        for move in get_available_moves(board):
            board[move] = 'X'
            score = minimax(board, depth + 1, False, alpha, beta)
            board[move] = ' ' # backtrack
            
            best_score = max(score, best_score)
            alpha = max(alpha, score)
            if beta <= alpha:
                break # Alpha-Beta Pruning
        return best_score
    else:
        best_score = math.inf
        for move in get_available_moves(board):
            board[move] = 'O'
            score = minimax(board, depth + 1, True, alpha, beta)
            board[move] = ' ' # backtrack
            
            best_score = min(score, best_score)
            beta = min(beta, score)
            if beta <= alpha:
                break # Alpha-Beta Pruning
        return best_score

def best_move(board):
    best_score = -math.inf
    move = -1
    for m in get_available_moves(board):
        board[m] = 'X'
        score = minimax(board, 0, False, -math.inf, math.inf)
        board[m] = ' '
        if score > best_score:
            best_score = score
            move = m
    return move

def play_game():
    board = [' '] * 9
    print("--- Unbeatable Tic-Tac-Toe AI ---")
    print("You are 'O', AI is 'X'.")
    print("Board positions are 0-8, reading left to right, top to bottom:")
    print_board([str(i) for i in range(9)])

    while True:
        # Human turn
        try:
            move = int(input("Enter your move (0-8): "))
            if move < 0 or move > 8 or board[move] != ' ':
                print("Invalid move. Spot taken or out of bounds. Try again.")
                continue
        except ValueError:
            print("Invalid input. Please enter a number between 0 and 8.")
            continue
        
        board[move] = 'O'
        
        if check_winner(board, 'O'):
            print_board(board)
            print("You win! (This shouldn't happen...)")
            break
        if is_draw(board):
            print_board(board)
            print("It's a draw!")
            break

        # AI turn
        print("AI is calculating...")
        ai_move = best_move(board)
        board[ai_move] = 'X'
        print_board(board)

        if check_winner(board, 'X'):
            print("AI wins! Better luck next time.")
            break
        if is_draw(board):
            print("It's a draw!")
            break

if __name__ == "__main__":
    play_game()
