import numpy as np
import os.path
import pandas as pd
from tabulate import tabulate
import pickle
import seaborn as sns


def menu():
    print("\n\n***** Welcome To Employee Attrition Prediction System *****")
    print("\n\nPlease Choose Your Option")
    print("\n1. Choose Your File Path")
    print("2. Exit\n")


def import_from_csv():
    while True:
        csv_path = input(
            "\nPlease enter the path to the CSV file containing Employee data, or type 'Exit' to go back to Main Menu: ")
        if csv_path.lower() == "exit":
            break
        if not os.path.exists(csv_path):
            print("File not found, Please enter a valid file path.")
        else:
            try:
                df = pd.read_csv(csv_path, sep=",")
                print(
                    "\n\nData successfully loaded from CSV file.\n\nHere are the first five rows from the CSV file.\n\n")

                print(tabulate(df.head(),  headers='keys', showindex=False,
                               tablefmt='fancy_grid',  numalign="center", stralign="center"))
                predict_result(df, csv_path)
                break
            except Exception as e:
                print("An exception occurred: ", e)


def predict_result(df, csvFile_path):
    try:
        print(df.info())
        # Load the machine learning model
        MODEL_PATH = os.path.abspath(os.path.join(
            # EmployeeData_prediction.h5
            os.path.dirname(__file__), "models", "EmployeeData_prediction.h5"))

        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
            results = model.predict(df)

            df['attrition_prediction'] = results
            output_data = df
            print(
                "\n\nPredictions are generated successfully.\n\nHere are the first five rows from the prediction results.\n\n")

            print(tabulate(output_data.head(),  headers='keys', showindex=False,
                           tablefmt='fancy_grid',  numalign="center", stralign="center"))
            print("\n")
            if (csvFile_path is not None):
                # Get the directory and base name of the input file
                input_dirname = os.path.dirname(csvFile_path)
                input_basename = os.path.basename(csvFile_path)

                # Create the output file path
                output_filename = os.path.join(
                    input_dirname, f'{os.path.splitext(input_basename)[0]}-predictions.csv')

                # Write the output DataFrame to a CSV file
                output_data.to_csv(output_filename, index=False)
                print(f"\n\nPredictions saved to {output_filename}\n\n")

    except Exception as ex:
        print(ex)


if __name__ == "__main__":
    option = -1

    while (True):
        menu()
        try:
            option = int(input())
        except:
            print("Invalid input. Please try again.")

        if (option == 1):
            import_from_csv()
        elif (option == 2):
            break
        else:
            option = 0

print("\n\nThank You For Choosing Employee Attrition Prediction System. Have A Nice Day!\n\n")
