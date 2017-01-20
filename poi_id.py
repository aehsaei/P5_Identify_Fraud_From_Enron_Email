#!/usr/bin/python

import sys
import pickle
from time                       import time
from pandas                     import pandas as pd
from sklearn                    import tree, preprocessing
from sklearn.naive_bayes        import GaussianNB
from sklearn.ensemble           import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model       import LogisticRegression, LinearRegression, Lasso
from sklearn.model_selection    import GridSearchCV
from sklearn.pipeline           import Pipeline
from sklearn.feature_selection  import RFE
from sklearn.cross_validation   import train_test_split
from sklearn.metrics            import accuracy_score, recall_score, precision_score

sys.path.append("../tools/")
from feature_format             import featureFormat, targetFeatureSplit
from tester                     import test_classifier, dump_classifier_and_data


def create_features_list():
    """Task 1: Create a list of feature names.
    Assemble the list using each type available: financial, email, poi
    :rtype: features_list(list of feature names)
    """

    # financial feature names (represent a value in US dollars)
    financial_features_list = [ 'salary',
                                'deferral_payments',
                                'total_payments',
                                'loan_advances',
                                'bonus',
                                'restricted_stock_deferred',
                                'deferred_income',
                                'total_stock_value',
                                'expenses',
                                'exercised_stock_options',
                                'other',
                                'long_term_incentive',
                                'restricted_stock',
                                'director_fees' ]

    # email feature names (represent a number of email messages)
    email_features_list = [ 'to_messages',
                            'from_poi_to_this_person',
                            'from_messages',
                            'from_this_person_to_poi',
                            'shared_receipt_with_poi' ]

    ### Task 1: Select features
    # features list contains all feature names beginning with POI
    # (Person Of Interest, boolean, represented as integer)
    features_list = ['poi']
    features_list.extend(financial_features_list)
    features_list.extend(email_features_list)

    return features_list


def remove_outliers(data_dict):
    """Task 2: Remove outliers.
    Look through the data and spot any outliers or errors in the dataset.
    :rtype: data_dict(dictionary of feature values)
    """

    ### Create a temporary dictionary for sorting
    temp_data = data_dict

    ### Sort the list by salary (highest first)
    temp_data = sorted(temp_data.items(), key=lambda item: item[1]['salary'], reverse=True)

    ### Print the highest 3 salaries in the list (exclude 'NaN' values)
    ### Find any unusual elements listed at the top
    count = 0
    print "People with highest salaries:"
    for person in temp_data:
        if person[1]['salary'] != 'NaN' and count <= 2:
            print person[0]
            count += 1

    print "\n"

    ### Print names of people with 'NaN' values many of the features
    count = 0
    print "People with 18 or more NaN values:"
    for person in temp_data:
        for feature in person[1]:
            if person[1][feature] == 'NaN':
                count += 1
        if count >= 18:
            print person[0]
        count = 0

    ### Remove these elements from the dataset. They are not people of interest, but spreadsheet totals
    data_dict.pop("TOTAL", 0)
    data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)

    ### These people have at least 18 'NaN' values for the features so I will remove these since 'NaN' is converted to 0
    data_dict.pop("LOCKHART EUGENE E", 0)
    data_dict.pop("WROBEL BRUCE", 0)
    data_dict.pop("WHALEY DAVID A", 0)

    return data_dict


def create_new_features(features_list, data_dict):
    """Task 3: Create new features.
    Create new features to allow for better POI accuracy by the classifier algorithms
    :rtype: features_list(list of feature names), data_dict(dictionary of feature values)
    """

    for person in data_dict:

        # feature 1: total stock value / total payments
        # I suspect the poi had very large stock values relative to their salaries
        try:
            if (data_dict[person]['total_payments'] == 'NaN' or
                data_dict[person]['total_stock_value'] == 'NaN'):
                data_dict[person]['stock_to_payments_ratio'] = 0
            else:
                data_dict[person]['stock_to_payments_ratio'] = \
                    (float(data_dict[person]['total_stock_value']) /
                     float(data_dict[person]['total_payments']))
        except:
            data_dict[person]['stock_to_payments_ratio'] = 0

        # feature 2: fraction of poi emails sent / total emails sent
        # I suspect that poi's sent a large portion of the total emails to POI's
        try:
            if (data_dict[person]['from_this_person_to_poi'] == 'NaN' or
                data_dict[person]['to_messages'] == 'NaN'):
                data_dict[person]['poi_to_total_emails_sent_ratio'] = 0
            else:
                data_dict[person]['poi_to_total_emails_sent_ratio'] = \
                    (float(data_dict[person]['from_this_person_to_poi']) /
                     float(data_dict[person]['to_messages']))
        except:
            data_dict[person]['poi_to_total_emails_sent_ratio'] = 0

        # feature 3: fraction of poi emails received / total emails received
        # I suspect that poi's received a large portion of the total emails from POI's
        try:
            if (data_dict[person]['from_poi_to_this_person'] == 'NaN' or
                data_dict[person]['from_messages'] == 'NaN'):
                data_dict[person]['poi_to_total_emails_received_ratio'] = 0
            else:
                data_dict[person]['poi_to_total_emails_received_ratio'] = \
                    (float(data_dict[person]['from_poi_to_this_person']) /
                     float(data_dict[person]['from_messages']))
        except:
            data_dict[person]['poi_to_total_emails_received_ratio'] = 0


    # Add the new feature names to the feature names list
    features_list.append('stock_to_payments_ratio')
    features_list.append('poi_to_total_emails_sent_ratio')
    features_list.append('poi_to_total_emails_received_ratio')

    return features_list, data_dict


def scale_features(features):
    """Scale the feature values
    To make the features more comparable, use the Min/Max Scaler In Sklearn
    Parameters: features = numpy array of shape [n_samples, n_features], Training set
    :rtype: features = numpy array of shape [n_samples, n_features_new], Transformed array
    """

    # Create an instance of the min_max_scaler
    min_max_scaler = preprocessing.MinMaxScaler()

    # Transform the feature set
    return min_max_scaler.fit_transform(features)


def create_classifiers_list():
    """Create a list of classifiers
    Assemble the list using a range of different classifiers
    :rtype: classifiers
    """
    classifiers = {}

    # Decision Tree
    classifiers["Decision Tree"] = tree.DecisionTreeClassifier()

    # Naive Bayes
    classifiers["Naive Bayes"] = GaussianNB()

    # Random Forest
    classifiers["Random Forest"] = RandomForestClassifier()

    # Adaboost / Decision tree
    classifiers["AdaBoost"] = AdaBoostClassifier()

    # Logistic Regression
    classifiers["Logistic Regression"] = LogisticRegression()

    return classifiers


def test_classifiers(classifiers, features_train, features_test, labels_train, labels_test):
    """Tests classifiers and find accuracy, precision, and recall scores for each
    :return: None
    """

    classifier_results = []

    ### Loop through each classifier train, predict, calculate scores
    for name, clf in classifiers.items():
        clf.fit(features_train, labels_train)
        pred = clf.predict(features_test)
        classifier_results.append({'Classifier': name,
                                   'Score Accuracy': accuracy_score(labels_test, pred),
                                   'Score Precision': precision_score(labels_test, pred),
                                   'Score Recall': recall_score(labels_test, pred)})

    print pd.DataFrame(classifier_results)

    return


def feature_ranking(features_train, labels_train, features_list):
    """Rank the features
    Use Recursive feature elimination to rank each of the features by importance
    :rtype: None
    """
    # use linear regression as the model
    reg = LinearRegression()

    # rank all features and select 1
    rfe = RFE(reg, n_features_to_select=1)
    rfe.fit(features_train, labels_train)

    print "Features sorted by rank:"
    print pd.DataFrame(sorted(zip(rfe.ranking_, features_list)))
    return


def custom_scorer(estimator, X, y):
    """Custom scorer to optimize the GridSearchCV for precision and recall average
    This function will be used in conjuction with the GridSearchCV tuning to optimize
    :rtype: average of precision and recall scores
    """
    pred = estimator.predict(X)

    precision = precision_score(y, pred)
    recall = recall_score(y, pred)

    if precision >= 0.3 and recall >= 0.3:
        return (precision + recall) / 2
    else:
        return 0


def tune_classifier(classifier, features_train, features_test, labels_train, labels_test):
    """Tune the chosen classifier
    Tune a classifier using GridSearchCV and a range of parameters
    :rtype: None
    """

    ### Take the current time for training time length calculation
    t0 = time()

    ### Assign all the parameters to perform the grid search over
    params = {'max_iter': [100, 200, 300],
              'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag'],
              'fit_intercept': [True, False],
              'C': [1, 10, 100, 10**4, 10**6, 10**8, 10**10],
              'tol': [0.1, 0.01, 0.001, 0.0001, 0.00001],
              'class_weight':['balanced']}

    ### Grid search for tuning the classifier
    grid = GridSearchCV(estimator=classifier, param_grid=params, scoring=custom_scorer)
    grid.fit(features_train, labels_train)
    pred = grid.predict(features_test)

    ### Dump the grid output
    print "GridSearchCV Best Parameters:"
    print grid.best_params_
    print "Max Accuracy Score: %f" % accuracy_score(labels_test, pred)
    print "Max Precision Score %f" % precision_score(labels_test, pred)
    print "Max Recall Score %f" % recall_score(labels_test, pred)

    ### Print out the training time
    print "training and predicting time:", round(time() - t0, 3), "s"

    return


def main():
    """main
    :rtype: None
    """

    ### Task 1: Create a feature list
    features_list = create_features_list()

    ### Load the dictionary containing the dataset
    with open("final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)

    ### Task 2: Remove outliers
    data_dict = remove_outliers(data_dict)

    ### Task 3: Create new features
    features_list, data_dict = create_new_features(features_list, data_dict)

    ### Store to my_dataset for easy export below.
    my_dataset = data_dict

    ### Extract features and labels from dataset for local testing
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    ### Scale the features to standardize
    features = scale_features(features)

    ### Split the data into a training and testing set
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)

    ### Rank the features
    feature_ranking(features_train, labels_train, features_list)

    ### Create a list of classifiers
    classifiers = create_classifiers_list()

    ### Task 4: Try a varity of classifiers
    test_classifiers(classifiers, features_train, features_test, labels_train, labels_test)

    ### Task 5: Tune your classifier to achieve better than .3 precision and recall
    tune_classifier(LogisticRegression(), features_train, features_test, labels_train, labels_test)

    ### Pipeline the MinMaxScaler since the test_classifier repartitions the non-scaled data
    clf = Pipeline(steps=[("scaler", preprocessing.MinMaxScaler()),
                          ("clf", LogisticRegression(C= 1000000,
                                                     max_iter=100,
                                                     fit_intercept=True,
                                                     solver='lbfgs',
                                                     tol=0.01,
                                                     class_weight='balanced'))])

    ### Check the results using the test_classifier task
    test_classifier(clf, my_dataset, features_list)

    ### Task 6: Dump your classifier, dataset, and features_list so anyone can
    ### check your results. You do not need to change anything below, but make sure
    ### that the version of poi_id.py that you submit can be run on its own and
    ### generates the necessary .pkl files for validating your results.
    dump_classifier_and_data(clf, my_dataset, features_list)


if __name__ == '__main__':
    main()
