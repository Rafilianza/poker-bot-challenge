"""Run a frozen agent with the official SDK protocol."""
import os
import random
import sys
from macpoker.sdk import load_bot_from_file, run_bot
from macpoker.bots.builtin import BUILTINS, RandomBot

spec, seed = sys.argv[1:3]
random.seed(seed)
if spec.startswith('house:'):
    name = spec.split(':', 1)[1]
    bot = RandomBot(seed=seed) if name == 'random' else BUILTINS[name]()
else:
    os.chdir(os.path.dirname(os.path.abspath(spec)))
    bot = load_bot_from_file(spec)
run_bot(bot)
