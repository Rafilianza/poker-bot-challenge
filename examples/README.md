# Small runnable examples

These three simple policies were written for this release. They demonstrate the
SDK interface and do not aim to play strong poker. They need only the base
requirements. No pretrained weights or third-party model implementations are
included here.

Upload the `.py` files in the dashboard's **Model library**, then select them in
**Evaluate** and run a 10-hand benchmark. The SDK supplies the four house
opponents. To import from a terminal while the dashboard is running:

```bash
python -m arena.manage import "examples/*.py"
```

Start with CPU. You do not need PyTorch or a GPU for this example.
