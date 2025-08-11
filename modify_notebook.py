import json

with open('cancer prediction.ipynb', 'r') as f:
    notebook = json.load(f)

for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        if 'LogisticRegression' in source_str:
            new_source = []
            for line in cell['source']:
                new_line = line.replace('LogisticRegression', 'RandomForestClassifier')
                new_line = new_line.replace('linear_model', 'ensemble')
                new_source.append(new_line)
            cell['source'] = new_source

with open('cancer prediction.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
