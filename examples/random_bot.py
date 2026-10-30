"""Teaching example: randomly call or fold when facing a bet."""

import random

from macpoker import Bot


class MyBot(Bot):
    name = 'Example: random caller'

    def on_match_start(self, info):
        self.random = random.Random(42)

    def act(self, state):
        if not state.to_call:
            return state.check()
        if self.random.random() < 0.7:
            return state.call()
        return state.fold()
