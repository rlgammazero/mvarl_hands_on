import numpy as np
from gridworld import GridWorldWithPits

class CliffWalk(GridWorldWithPits):

    def __init__(self, proba_succ=0.95):
        grid1 = [
            ['', '', '', '', '', '', '', '', '', '', '', '',],
            ['', '', '', '', '', '', '', '', '', '', '', '',],
            ['', '', '', '', '', '', '', '', '', '', '', '',],
            ['s', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'g']
        ]
        grid1_MAP = [
            "+-----------------------+",
            "| : : : : : : : : : : : |",
            "| : : : : : : : : : : : |",
            "| : : : : : : : : : : : |",
            "|S:x:x:x:x:x:x:x:x:x:x:G|",
            "+-----------------------+",
        ]
        super(CliffWalk, self).__init__(grid=grid1, txt_map=grid1_MAP, proba_succ=proba_succ, uniform_trans_proba=0)

if __name__ == '__main__':
    env = CliffWalk()
