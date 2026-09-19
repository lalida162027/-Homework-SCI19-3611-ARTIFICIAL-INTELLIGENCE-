import math
import tkinter as tk
from tkinter import messagebox

# Hexapawn Config

ROWS = 3
COLS = 3

EMPTY = "."
HUMAN = "X"
AI = "O"


def initial_board():
    return [
        ["O", "O", "O"],  # AI
        [".", ".", "."],
        ["X", "X", "X"]   # Human
    ]


# Game Logic

def get_moves(board, player):
    moves = []
    direction = -1 if player == HUMAN else 1
    opponent = AI if player == HUMAN else HUMAN

    for r in range(ROWS):
        for c in range(COLS):
            if board[r][c] != player:
                continue

            nr = r + direction

            # เดินตรง
            if 0 <= nr < ROWS and board[nr][c] == EMPTY:
                moves.append(((r, c), (nr, c)))

            # กินซ้าย
            if 0 <= nr < ROWS and c - 1 >= 0 and board[nr][c - 1] == opponent:
                moves.append(((r, c), (nr, c - 1)))

            # กินขวา
            if 0 <= nr < ROWS and c + 1 < COLS and board[nr][c + 1] == opponent:
                moves.append(((r, c), (nr, c + 1)))

    return moves


def make_move(board, move):
    new_board = [row[:] for row in board]
    (r1, c1), (r2, c2) = move
    new_board[r2][c2] = new_board[r1][c1]
    new_board[r1][c1] = EMPTY
    return new_board


def winner(board):
    for c in range(COLS):
        if board[0][c] == HUMAN:
            return HUMAN
    for c in range(COLS):
        if board[ROWS - 1][c] == AI:
            return AI

    if not any(HUMAN in row for row in board):
        return AI
    if not any(AI in row for row in board):
        return HUMAN

    if len(get_moves(board, HUMAN)) == 0:
        return AI
    if len(get_moves(board, AI)) == 0:
        return HUMAN

    return None


def evaluate(board):
    result = winner(board)
    if result == AI:
        return 100
    if result == HUMAN:
        return -100

    ai_count = sum(row.count(AI) for row in board)
    human_count = sum(row.count(HUMAN) for row in board)
    return (ai_count - human_count) * 10


def alpha_beta(board, depth, alpha, beta, maximizing):
    result = winner(board)
    if result == AI:
        return 100 + depth
    if result == HUMAN:
        return -100 - depth
    if depth == 0:
        return evaluate(board)

    if maximizing:
        value = -math.inf
        for move in get_moves(board, AI):
            new_board = make_move(board, move)
            value = max(value, alpha_beta(new_board, depth - 1, alpha, beta, False))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = math.inf
        for move in get_moves(board, HUMAN):
            new_board = make_move(board, move)
            value = min(value, alpha_beta(new_board, depth - 1, alpha, beta, True))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value


def alpha_beta_player(board, depth=10):
    moves = get_moves(board, AI)
    if not moves:
        return None

    best_move = moves[0]
    best_value = math.inf

    for move in moves:
        new_board = make_move(board, move)
        value = alpha_beta(new_board, depth - 1, -math.inf, math.inf, False)
        
        # เลือกค่าที่น้อยที่สุด (กลายเป็นการเดินที่แย่สำหรับ AI)
        if value < best_value:
            best_value = value
            best_move = move

    return best_move


# GUI Application

class HexapawnGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Hexapawn Game (You Always Win)")
        self.board = initial_board()
        self.selected = None

        self.status_label = tk.Label(root, text="ตาของคุณ (X): คลิกเลือกหมากแล้วคลิกช่องเดิน", font=("Arial", 12))
        self.status_label.pack(pady=10)

        self.frame = tk.Frame(root)
        self.frame.pack()

        self.buttons = [[None for _ in range(COLS)] for _ in range(ROWS)]
        for r in range(ROWS):
            for c in range(COLS):
                btn = tk.Button(self.frame, text="", font=("Arial", 28, "bold"), width=4, height=2,
                                command=lambda row=r, col=c: self.on_square_click(row, col))
                btn.grid(row=r, column=c, padx=3, pady=3)
                self.buttons[r][c] = btn

        self.update_board_ui()

    def update_board_ui(self):
        for r in range(ROWS):
            for c in range(COLS):
                piece = self.board[r][c]
                text = ""
                bg = "#f0f0f0"

                if piece == HUMAN:
                    text = "X"
                    bg = "#d0e8ff"
                elif piece == AI:
                    text = "O"
                    bg = "#ffd0d0"

                if self.selected == (r, c):
                    bg = "#ffff99"

                self.buttons[r][c].config(text=text, bg=bg)

    def on_square_click(self, r, c):
        res = winner(self.board)
        if res is not None:
            return

        if self.selected is None:
            if self.board[r][c] == HUMAN:
                self.selected = (r, c)
                self.update_board_ui()
        else:
            r1, c1 = self.selected
            r2, c2 = r, c
            move = ((r1, c1), (r2, c2))

            legal_moves = get_moves(self.board, HUMAN)

            if move in legal_moves:
                self.board = make_move(self.board, move)
                self.selected = None
                self.update_board_ui()

                res = winner(self.board)
                if res is not None:
                    self.end_game(res)
                    return

                self.status_label.config(text="AI กำลังเดินพลาด...")
                self.root.after(400, self.ai_turn)
            else:
                if self.board[r][c] == HUMAN:
                    self.selected = (r, c)
                else:
                    self.selected = None
                self.update_board_ui()

    def ai_turn(self):
        move = alpha_beta_player(self.board, depth=10)
        if move is None:
            self.end_game(HUMAN)
            return

        self.board = make_move(self.board, move)
        self.update_board_ui()

        res = winner(self.board)
        if res is not None:
            self.end_game(res)
        else:
            self.status_label.config(text="ตาของคุณ (X): คลิกเลือกหมากแล้วคลิกช่องเดิน")

    def end_game(self, res):
        for r in range(ROWS):
            for c in range(COLS):
                self.buttons[r][c].config(state="disabled")

        if res == HUMAN:
            self.status_label.config(text="คุณชนะแล้ว!")
            messagebox.showinfo("จบเกม", "ยินดีด้วย คุณชนะแล้ว!")
        else:
            self.status_label.config(text="AI ชนะ!")
            messagebox.showinfo("จบเกม", "AI ชนะ!")


if __name__ == "__main__":
    root = tk.Tk()
    app = HexapawnGUI(root)
    root.mainloop()