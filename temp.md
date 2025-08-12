# Tensor test
```bash
source .venv/bin/activate && xmake && xmake install && uv pip install ./python/ && python test/test_tensor.py
```
# Operators test
Add
```bash
source .venv/bin/activate && xmake && xmake install && uv pip install ./python/ && python test/ops/add.py
```
Argmax
```bash
source .venv/bin/activate && xmake && xmake install && uv pip install ./python/ && python test/ops/argmax.py
```
embedding
```bash
source .venv/bin/activate && xmake && xmake install && uv pip install ./python/ && python test/ops/embedding.py
```