# Creating a Gridworld game

from Gridworld import Gridworld

game = Gridworld(size=4, mode='static')
print(game.display())

game.makeMove('d')
game.makeMove('d')
game.makeMove('l')
print(game.display())
print(game.reward())
print(game.board.render_np())
print(game.board.render_np().shape)
