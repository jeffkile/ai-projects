import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """

    evidence = []
    labels = []

    with open(filename, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        first_row_found = False
        for row in spamreader:
            if first_row_found == False:
                first_row_found = True
                continue
            vars = row[0].split(',')

            evidence_row = []

        # - Administrative, an integer
            evidence_row.append(int(vars[0]))
        #- Administrative_Duration, a floating point number
            evidence_row.append(float(vars[1]))
        # - Informational, an integer
            evidence_row.append(int(vars[2]))
        #- Informational_Duration, a floating point number
            evidence_row.append(float(vars[3]))
        #- ProductRelated, an integer
            evidence_row.append(int(vars[4]))
        #- ProductRelated_Duration, a floating point number
            evidence_row.append(float(vars[5]))
        # - BounceRates, a floating point number
            evidence_row.append(float(vars[6]))
        # - ExitRates, a floating point number
            evidence_row.append(float(vars[7]))
        # - PageValues, a floating point number
            evidence_row.append(float(vars[8]))
        # - SpecialDay, a floating point number
            evidence_row.append(float(vars[9]))
        # - Month, an index from 0 (January) to 11 (December)
            evidence_row.append(month_to_num(vars[10]))
        # - OperatingSystems, an integer
            evidence_row.append(int(vars[11]))
        # - Browser, an integer
            evidence_row.append(int(vars[12]))
        # - Region, an integer
            evidence_row.append(int(vars[13]))
        # - TrafficType, an integer
            evidence_row.append(int(vars[14]))
        # - VisitorType, an integer 0 (not returning) or 1 (returning)
            evidence_row.append(1 if vars[15] == "Returning_Visitor" else 0)
        # - Weekend, an integer 0 (if false) or 1 (if true)
            evidence_row.append(1 if vars[16] == "TRUE" else 0)

            evidence.append(evidence_row)
            labels.append(1 if vars[17] == "TRUE" else 0)

    return evidence, labels

def month_to_num(month):
    if month == "Jan":
        return 0
    if month == "Feb":
        return 1
    if month == "Mar":
        return 2
    if month == "Apr":
        return 3
    if month == "May":
        return 4
    if month == "June":
        return 5
    if month == "Jul":
        return 6
    if month == "Aug":
        return 7
    if month == "Sep":
        return 8
    if month == "Oct":
        return 9
    if month == "Nov":
        return 10
    if month == "Dec":
        return 11

def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """

    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(evidence, labels)

    return neigh

def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """

    positive_total_count = 0
    positive_accurate_count = 0
    negative_total_count = 0
    negative_accurate_count = 0
    for index in range(0, len(labels)):
        if labels[index] == 0:
            negative_total_count += 1
            if predictions[index] == 0:
                negative_accurate_count += 1
        else:
            positive_total_count += 1
            if predictions[index] == 1:
                positive_accurate_count += 1

    sensitivity =  positive_accurate_count / positive_total_count if positive_total_count > 0 else 0
    specificity =  negative_accurate_count / negative_total_count if negative_total_count > 0 else 0

    return (sensitivity, specificity)




if __name__ == "__main__":
    main()
