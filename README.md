# Immo Eliza Project - Part 3: Regression

## 1. Description:

(tbd)

## 2. Installation:
1. Clone the Repository: Start by cloning this GitHub repository to your local machine using the following command:

    `git clone https://github.com/brammichielsen/challenge-regression.git`

2. Create and activate a Virtual Environment: change into the project directory and create a virtual environment to isolate the project's dependencies.

3. Install Dependencies: With the virtual environment activated, install the required dependencies from the requirements.txt file. This will ensure you have the same versions of the dependencies used in this project:

    `pip install -r requirements.txt`

    This will install pandas, scikit_learn, and xgboost with the specific versions specified in the requirements.txt file.

4. Run the Project: Now that the dependencies are installed, you can run the project by executing the main Python script.

    `python main.py`

## 3. Usage:

(tbd) 

## 4. Folder structure:

└── challenge-regression/
    ├── data/
    │   └── holds the .csv file the program will use for input
    ├── src/
    │   └── data_format.py
    │   └── data_import.py
    │   └── data_prepare.py
    │   └── model_eval.py
    │   └── model_select_train.py
    └── .gitignore
    └── main.py
    └── README.md
    └── requirements.txt

## 5. Contributors:
This was a solo project by Bram Michielsen.
- https://github.com/brammichielsen
- https://www.linkedin.com/in/brammichielsen/

## 6. Timeline:
- Day 1: 
    - created initial setup (virtual environment, packages, updated repository)
    - worked on logical flow and built initial script/.py file and folder structure
- Day 2: 
    - took a step back to build pipeline in Jupyter notebook for debugging and overview purposes
- Day 3:
    - completed one-pass, start-to-finish, linear regression pipeline in Jupyter notebook
    - started integration of non-linear regression pipeline using xgboost for score comparison
- Day 4: 
    - finished combining linear regression and non-linear regression flows
    - integrated all the Jupyter Notebook modules into the corresponding .py files
    - added the .csv encapsulation, using regular expression to read the .csv file in /data
    - ensured all the functions have docstrings
    - touched up code comments
    - wrote the README.md

## 7. Personal situation:
Working on this third part of the project helped me face my personal pitfalls of perfectionism and resulting fear of failure. It made me focus on pushing through to a functioning MVP while retaining command of the code and logic, before getting bogged down in the details. 

Taking a step back to create the separate modules of the program in individual Jupyter Notebook cells really helped with keeping the overall structure in mind, as well as incrementally building out functions and debugging.