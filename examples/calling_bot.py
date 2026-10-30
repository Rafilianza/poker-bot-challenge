"""Teaching example written for this project: call or check every turn."""

from macpoker import Bot


class MyBot(Bot):
    name = 'Example: caller'

    def act(self, state):
        return state.call() if state.to_call else state.check()
