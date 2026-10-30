"""Teaching example: pay at most two chips to continue; otherwise fold."""

from macpoker import Bot


class MyBot(Bot):
    name = 'Example: cautious'

    def act(self, state):
        if not state.to_call:
            return state.check()
        if state.to_call <= 2:
            return state.call()
        return state.fold()
