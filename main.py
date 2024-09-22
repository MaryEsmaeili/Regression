# main.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from logistic_regression import LogisticRegression

def main():
    # Import data from chemBL
    from chembl_webresource_client.new_client import new_client
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    # Initialize the ChEMBL client
    chembl = new_client

    target_chembl_id = 'CHEMBL2034'  # Glucocorticoid receptor targets

    # Get bioactivity data for the target
    activity = chembl.activity.filter(target_chembl_id=target_chembl_id, standard_type='IC50')

    # Convert to DataFrame
    df_activity = pd.DataFrame(activity)

    # Select relevant columns and drop rows with missing values
    df_activity = df_activity[['molecule_chembl_id', 'canonical_smiles', 'standard_value']]
    df_activity.dropna(inplace=True)

    # Convert IC50 values to binary labels (active if IC50 < 1000 nM, inactive otherwise)
    df_activity['bioactivity_label'] = df_activity['standard_value'].apply(lambda x: 1 if float(x) < 1000 else 0)

    # Function to calculate molecular descriptors for smiles
    def calculate_descriptors(smiles):
        mol = Chem.MolFromSmiles(smiles)
        descriptors = {
            'MolecularWeight': Descriptors.MolWt(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'TPSA': Descriptors.TPSA(mol)
        }
        return pd.Series(descriptors)

    # Calculate descriptors for each molecule
    df_descriptors = df_activity['canonical_smiles'].apply(calculate_descriptors)
    df_GR = pd.concat([df_descriptors, df_activity], axis=1)

    # Drop unnecessary columns (standard_value, molecule_chembl_id)
    df_GR.drop(columns=['standard_value', 'molecule_chembl_id', 'canonical_smiles'], inplace=True)

    # Features and target variable
    X = df_GR.drop(columns=['bioactivity_label']).values  # Molecular descriptors
    y = df_GR['bioactivity_label'].values  # Bioactivity labels

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create an instance of the LogisticRegression class
    log_reg = LogisticRegression(learning_rate=0.01, num_iters=15000, lambda_=0.7)

    # Train the model
    log_reg.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = log_reg.predict(X_test)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

# Ensure the script runs when called directly
if __name__ == "__main__":
    main()
