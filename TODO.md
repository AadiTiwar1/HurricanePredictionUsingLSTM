\***\*Folders:\*\***

`src/`- Core code/classes necessary for the project (this is code that generally defines processes)

`scripts` - Utility scripts for project and application setup (this is code that generally executes processes, these scripts will be called by the .project-metadata.yaml file)

`app/` - Assets needed to support the front end application (this is code that purely supports the front-end). The linked example goes along with the

`static/` - Any images referenced in project docs

\***\*Files:\*\***

`README.md` - Should fully describe your project including: overview of your AMP, a brief overview of any novel ML techniques used, an explanation of the project structure, and instructions to run your AMP.

\***\*Needs Testing:\*\***

`train.py` - script to prep data and train and/or validate model

`static/dataset/atlantic.csv` - figure out how to drop columns and clean up the file AFTER downloading with download_data.py

`src/model.py` - figure out how to create a prediction method that takes in inputs from the frontend `app/app.py` and makes a prediction that can be displayed in the frontend
