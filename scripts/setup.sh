# Link local repository to git
git init
git add *
git add .gitignore
git commit -m 'Initialized the repository'
git remote add origin git@github.com:tobifinn/ensemble_transformer.git
git push -u origin master
