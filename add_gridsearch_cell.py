import json

with open('cancer prediction.ipynb', 'r') as f:
    notebook = json.load(f)

# Remove the old grid search cell if it exists
notebook['cells'] = [c for c in notebook['cells'] if c.get('id') != 'grid-search-cell']

new_cell_source = [
    "from sklearn.model_selection import GridSearchCV\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn import metrics\n",
    "\n",
    "param_grid = {\n",
    "    'n_estimators': [100, 200, 300],\n",
    "    'max_features': ['sqrt', 'log2'],\n",
    "    'max_depth' : [4,5,6,7,8],\n",
    "    'criterion' :['gini', 'entropy']\n",
    "}\n",
    "\n",
    "grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=55), param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)\n",
    "grid_search.fit(X_train, y_train)\n",
    "\n",
    "print(\"Best parameters found: \", grid_search.best_params_)\n",
    "y_pred_grid = grid_search.predict(X_test)\n",
    "print(\"Accuracy with tuned parameters: \", metrics.accuracy_score(y_test, y_pred_grid))"
]

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "grid-search-cell-v2",
    "metadata": {},
    "outputs": [],
    "source": new_cell_source
}

notebook['cells'].append(new_cell)

with open('cancer prediction.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
