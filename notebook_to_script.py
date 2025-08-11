import json

with open('cancer prediction.ipynb', 'r') as f:
    notebook = json.load(f)

python_code = ''
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        python_code += ''.join(cell['source']) + '\n'

with open('run_notebook.py', 'w') as f:
    f.write(python_code)
