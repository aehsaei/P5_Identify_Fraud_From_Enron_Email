#!/usr/bin/python

import sys
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from time                       import time
from pandas                     import pandas as pd
from sklearn                    import tree, preprocessing
from sklearn.naive_bayes        import GaussianNB
from sklearn.ensemble           import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model       import LogisticRegression, LinearRegression, Lasso
from sklearn.model_selection    import GridSearchCV
from sklearn.pipeline           import Pipeline
from sklearn.feature_selection  import RFE, SelectKBest
from sklearn.cross_validation   import train_test_split, StratifiedShuffleSplit
from sklearn.metrics            import accuracy_score, recall_score, precision_score, make_scorer
from matplotlib                 import colors
from matplotlib.colors          import ListedColormap
from sklearn.svm                import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.append("../tools/")
from feature_format             import featureFormat, targetFeatureSplit
from tester                     import test_classifier, dump_classifier_and_data


def create_features_list():
    """Task 1: Create a list of feature names.
    Assemble the list using each type available: financial, email, poi
    :rtype: features_list(list of feature names)
    """

    print "create_features_list():: Create a list of features"

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

    ### Create a list of features starting with 'poi'
    features_list = ['poi']
    features_list.extend(financial_features_list)
    features_list.extend(email_features_list)

    print features_list
    print "..."

    return features_list


def remove_outliers(data_dict):
    """Remove outliers
    Look through the data and spot any outliers or errors in the dataset.
    :rtype: data_dict(dictionary of feature values)
    """

    print "remove_outliers():: Remove Outliers from the dataset"

    ### Create a temporary dictionary for sorting
    temp_data = dict(data_dict)

    ### Sort the list by salary (highest first)
    temp_data = sorted(temp_data.items(), key=lambda item: item[1]['salary'], reverse=True)

    ### Print the highest 3 salaries in the list (exclude 'NaN' values)
    ### Find any unusual elements listed at the top
    count = 0
    print "People with highest salaries:"
    for person in temp_data:
        if person[1]['salary'] != 'NaN' and count <= 2:
            print "  ", person[0], ",", person[1]['salary']
            count += 1

    ### Print names of people with 'NaN' values many of the features
    count = 0
    print "People with 18 or more NaN values:"
    for person in temp_data:
        for feature in person[1]:
            if person[1][feature] == 'NaN':
                count += 1
        if count >= 18:
            print "  ", person[0], ",", count, "NaN values"
        count = 0

    ### Remove these elements from the dataset. They are not people of interest, but spreadsheet totals
    data_dict.pop("TOTAL", 0)
    data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)

    ### These people have at least 18 'NaN' values for the features so I will remove these since 'NaN' is converted to 0
    data_dict.pop("LOCKHART EUGENE E", 0)
    data_dict.pop("WROBEL BRUCE", 0)
    data_dict.pop("WHALEY DAVID A", 0)

    ### Find total payments by POI's
    total_payments_poi = 0
    total_payments_non_poi = 0
    total_poi = 0
    total_non_poi = 0
    for person in temp_data:
        if (person[1]['total_payments'] != 'NaN'):
            if (person[1]['poi']):
                total_payments_poi += person[1]['total_payments']
                total_poi += 1
            else:
                total_payments_non_poi += person[1]['total_payments']
                total_non_poi += 1

    total_payments = []
    total_payments.append({'POI': 'POI', 'Average_total_payments': total_payments_poi/total_poi})
    total_payments.append({'POI': 'NON POI', 'Average_total_payments': total_payments_non_poi/total_non_poi})

    print total_payments

    # Plot total payments bar plot
    total_payments_df = pd.DataFrame(total_payments)
    sns.set_style("whitegrid")
    ax = sns.barplot(total_payments_df['POI'], total_payments_df['Average_total_payments'], palette="Set3")
    ax.set(xlabel='', ylabel='Average Total Payments (USD)')
    plt.tight_layout()
    plt.savefig('total_payments.png')
    plt.close()

    print "..."

    return data_dict


def create_new_features(features_list, data_dict):
    """Create new features
    Create new features to allow for better POI accuracy by the classifier algorithms
    :rtype: features_list(list of feature names), data_dict(dictionary of feature values)
    """

    print "create_new_features():: Create new features and add to features list"

    # Loop through all people in the data
    for person in data_dict:

        # feature 1: total stock value / total payments
        # I suspect the poi had very large stock values relative to their salaries
        try:
            # Catch case where either value is 'NaN'
            if (data_dict[person]['total_payments'] == 'NaN' or
                data_dict[person]['total_stock_value'] == 'NaN'):
                data_dict[person]['stock_to_payments_ratio'] = 0
            # Calculate ratio for new feature
            else:
                data_dict[person]['stock_to_payments_ratio'] = \
                    (float(data_dict[person]['total_stock_value']) /
                     float(data_dict[person]['total_payments']))
        except:
            data_dict[person]['stock_to_payments_ratio'] = 0

        # feature 2: fraction of poi emails sent / total emails sent
        # I suspect that poi's sent a large portion of the total emails to POI's
        #REMOVE# try:
        #REMOVE#    # Catch case where either value is 'NaN'
        #REMOVE#    if (data_dict[person]['from_this_person_to_poi'] == 'NaN' or
        #REMOVE#        data_dict[person]['to_messages'] == 'NaN'):
        #REMOVE#        data_dict[person]['poi_to_total_emails_sent_ratio'] = 0
        #REMOVE#    # Calculate ratio for new feature
        #REMOVE#    else:
        #REMOVE#        data_dict[person]['poi_to_total_emails_sent_ratio'] = \
        #REMOVE#            (float(data_dict[person]['from_this_person_to_poi']) /
        #REMOVE#             float(data_dict[person]['to_messages']))
        #REMOVE#except:
        #REMOVE#    data_dict[person]['poi_to_total_emails_sent_ratio'] = 0

        # feature 3: fraction of poi emails received / total emails received
        # I suspect that poi's received a large portion of the total emails from POI's
        #REMOVE#try:
        #REMOVE#    # Catch case where either value is 'NaN'
        #REMOVE#    if (data_dict[person]['from_poi_to_this_person'] == 'NaN' or
        #REMOVE#        data_dict[person]['from_messages'] == 'NaN'):
        #REMOVE#        data_dict[person]['poi_to_total_emails_received_ratio'] = 0
        #REMOVE#    # Calculate ratio for new feature
        #REMOVE#    else:
        #REMOVE#        data_dict[person]['poi_to_total_emails_received_ratio'] = \
        #REMOVE#            (float(data_dict[person]['from_poi_to_this_person']) /
        #REMOVE#             float(data_dict[person]['from_messages']))
        #REMOVE#except:
        #REMOVE#    data_dict[person]['poi_to_total_emails_received_ratio'] = 0

    # Add the new feature names to the feature names list
    features_list.append('stock_to_payments_ratio')
    #REMOVE#features_list.append('poi_to_total_emails_sent_ratio')
    #REMOVE#features_list.append('poi_to_total_emails_received_ratio')

    print features_list
    print "..."
    return features_list, data_dict


def scale_features(features):
    """Scale the feature values
    To make the features more comparable, use the Min/Max Scaler In Sklearn
    Parameters: features = numpy array of shape [n_samples, n_features], Training set
    :rtype: features = numpy array of shape [n_samples, n_features_new], Transformed array
    """

    print "scale_features():: Scale features using the MinMaxScaler"

    # Create an instance of the min_max_scaler
    min_max_scaler = preprocessing.MinMaxScaler()

    print "..."

    # Scale the features using the min_max_scaler
    return min_max_scaler.fit_transform(features)


def create_classifiers_list():
    """Create a list of classifiers
    Assemble the list using a range of different classifiers
    :rtype: classifiers
    """

    print "create_classifiers_list():: Create a list of classifiers to test"

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

    # Linear Support Vector Classification
    classifiers["Linear SVC"] = LinearSVC()

    print classifiers.keys()
    print "..."

    return classifiers


def test_classifiers(classifiers, dataset, features_list):
    """Tests classifiers and find accuracy, precision, and recall scores for each
    :return: None
    """

    print "test_classifiers():: Test each classifier in the list of classifiers"

    classifier_results = []
    clf_scores = []

    ### Loop through each classifier train, predict, calculate scores
    for name, clf in classifiers.items():
        clf_scores = get_classifier_scores(clf, dataset, features_list)
        clf_scores.update({'Classifier': name})
        classifier_results.append(clf_scores)

    print pd.DataFrame(classifier_results)
    print "..."

    return


def rank_features(clf, features, labels, features_list):
    """Rank the features
    Use Recursive feature elimination to rank each of the features by importance
    :rtype: List of dictionary values containing the rank of each feature
    """

    print "rank_features():: Rank each feature using Recursive Feature Elimination"

    feature_rank_results = []

    # Rank all features and select 1
    rfe = RFE(clf, n_features_to_select=1)
    rfe.fit(features, labels)

    # Loop through the features and make a list of feature ranks
    if features_list[0] == 'poi':
        for index, feature in enumerate(features_list[1:]):
            feature_rank_results.append({'feature': feature, 'rank': rfe.ranking_[index]})
    else:
        for index, feature in enumerate(features_list):
            feature_rank_results.append({'feature': feature, 'rank': rfe.ranking_[index]})

    return feature_rank_results


def get_classifier_scores(clf, my_dataset, features_list, folds=1000):
    """Get Classifier Scores

    :rtype: List of dictonary values containing the number of features,
    accuracy, precision, recall scores for a classifier
    """

    print "get_classifier_scores():: Calculate the average classifier score using multiple folds"
    print clf

    data = featureFormat(my_dataset, features_list, sort_keys=True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    true_negatives  = 0
    false_negatives = 0
    true_positives  = 0
    false_positives = 0
    for train_idx, test_idx in cv:
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )

        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        return {'features': len(features_list),
                'precision': precision,
                'recall': recall}
    except:
        print "Got a divide by zero when trying out:", clf
        print "Precision or recall may be undefined due to a lack of true positive predicitons."


def test_features(clf, features, labels, dataset, features_list):
    """Test Features
    Use SelectKBest and loop through all features to find importance of each
    prints out the SelectKBest feature scores
    prints out the accuracy,precision,recall scores for numbers of features
    prints out the RFE feature rank scores
    :rtype: None
    """
    feature_amount_results = []
    feature_score_results = []

    print "Testing Features using", clf

    # Grab all features except 'poi', make sure to create a separate copy so original isn't changed
    temp_features_list = list(features_list)
    temp_features_list.remove('poi')

    ### Loop through numbers of features
    for index in range(1, len(temp_features_list)):

        print "Testing", index, "/", len(temp_features_list), "Features"

        ### Use SelectKBest to select best features
        pipeline = Pipeline(steps=[("sel", SelectKBest(k=index)),
                                   ("clf", clf)])

        ### Get classifier score
        classifier_score = get_classifier_scores(pipeline, dataset, features_list)

        ### Fix number of features since using kbest
        classifier_score['features'] = index
        feature_amount_results.append(classifier_score)

    ### Calculate the importance of each feature using SelectKBest
    for index, feature in enumerate(temp_features_list):
        selectbest = SelectKBest(k='all')
        selector = selectbest.fit(features, labels)
        feature_score_results.append({'feature': feature, 'score': selector.scores_[index]})

    feature_score_results_df = pd.DataFrame(feature_score_results)
    feature_amount_results_df = pd.DataFrame(feature_amount_results)
    feature_rank_results_df = pd.DataFrame(rank_features(clf, features, labels, temp_features_list))

    # Print the feature rank results
    print "Feature Rank Results:"
    print feature_rank_results_df.sort('rank', ascending=1)

    # Number of features affect on precision and recall scores
    feature_amount_results_df = feature_amount_results_df.sort('features', ascending=1)
    feature_amount_results_df.set_index('features')

    ### Line plot the Number of Features -> Precision and Recall Scores
    plt.plot(feature_amount_results_df['features'],
             feature_amount_results_df['precision'],
             marker='o',
             label='Precision Score')
    plt.plot(feature_amount_results_df['features'],
             feature_amount_results_df['recall'],
             marker='o',
             label='Recall Score')
    plt.legend(loc='lower right')
    plt.xlabel('Feature')
    plt.ylabel('SelectKBest Score')
    plt.savefig('feature_amount_results.png')
    plt.close()
    print "Feature Amount Results:"
    print feature_amount_results_df

    # Barplot of each feature Kbest score
    feature_score_results_df = feature_score_results_df.sort('score', ascending=False)
    ax = sns.barplot(feature_score_results_df['feature'],
                     feature_score_results_df['score'])
    ax.set(xlabel='Feature', ylabel='SelectKBest Score')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('feature_score_results.png')
    plt.close()

    print "..."

    return


def test_new_features(clf, features, labels, dataset, features_list):
    """Test Features
    prints out the accuracy,precision,recall scores for numbers of features
    :rtype: None
    """
    new_feature_scores = []

    print "Testing New Features using", clf

    # Grab all features
    temp_features_list = list(features_list)

    # Remove the new features
    temp_features_list.remove('stock_to_payments_ratio')
    temp_features_list.remove('poi_to_total_emails_sent_ratio')
    temp_features_list.remove('poi_to_total_emails_received_ratio')

    # Baseline score
    classifier_score = get_classifier_scores(clf, dataset, temp_features_list)
    new_feature_scores.append({"Baseline Score": classifier_score})

    # Loop through numbers of features
    for feature in ['stock_to_payments_ratio',
                    'poi_to_total_emails_sent_ratio',
                    'poi_to_total_emails_received_ratio']:

        print "Testing classifier score without %s feature" % feature

        # Add one of the new features
        temp_features_list.append(feature)

        # Get classifier score
        classifier_score = get_classifier_scores(clf, dataset, temp_features_list)
        new_feature_scores.append({feature: classifier_score})

        # Remove feature
        temp_features_list.remove(feature)

    print new_feature_scores
    print "..."

    return


def custom_score(y, y_pred):
    """Custom scorer to optimize the GridSearchCV for precision and recall average
    This function will be used in conjuction with the GridSearchCV tuning to optimize
    :rtype: combined precision and recall scores
    """

    ### Calculate the Precision Score
    precision = precision_score(y, y_pred)

    ### Calculate the Recall Score
    recall = recall_score(y, y_pred)

    ### Return the total precision and recall score only if both over threshold
    if precision >= 0.3 and recall >= 0.3:
        return (precision + recall)
    else:
        return 0


def tune_classifier(clf, features, labels):
    """Tune the chosen classifier
    Tune a classifier using GridSearchCV and a range of parameters
    :rtype: Dictionary of best parameter values
    """

    print "tune_classifier():: Use GridSearchCV to find optimal classifier parameters"

    ### Take the current time for training time length calculation
    t0 = time()

    clf = Pipeline(steps=[("sel", SelectKBest()),
                          ("clf", clf)])

    print clf.get_params()

    # Set the parameters
    params = [{'sel__k': [4, 16, 'all'],
               'clf__n_estimators': [1, 100],
               'clf__learning_rate': [1, 10, 50],
               'clf__algorithm': ['SAMME', 'SAMME.R']}]

    ### Setup a cross validation object
    cv = StratifiedShuffleSplit(labels, 100, random_state=42)

    ### Create the custom scorer function
    custom_scorer = make_scorer(custom_score, greater_is_better=True)

    ### Grid search for tuning the classifier
    grid = GridSearchCV(estimator=clf, param_grid=params, cv=cv, scoring=custom_scorer)

    ### Fit the grid to the training set
    grid.fit(features, labels)

    ### Print out the best parameters
    print("Best parameters set found on development set:")
    print(grid.best_params_)

    ### Print out the training time
    print "Time to run the GridSearchCV:", round(time() - t0, 3), "s"
    print "..."

    return grid.best_params_


def visual_gridsearch(model, X, y):
    """Create a visual gridsearch figure
    Create a heatmap of two parameters tuned using GridSearchCV
    :rtype: None
    """
    print "visual_gridsearch():: Use GridSearchCV to create a visual heatmap"

    ddl_heat = ['#DBDBDB', '#DCD5CC', '#DCCEBE', '#DDC8AF', '#DEC2A0', '#DEBB91', \
                '#DFB583', '#DFAE74', '#E0A865', '#E1A256', '#E19B48', '#E29539']
    ddlheatmap = colors.ListedColormap(ddl_heat)

    ### Define the parameter value ranges
    n_estimator_range = [1, 100, 200]
    learning_rate_range = [1, 2, 10, 50]

    ### Combine the parameters into a dictionary
    param_grid = dict(clf__n_estimators = n_estimator_range,
                      clf__learning_rate = learning_rate_range)

    ### Setup a cross validation object
    cv = StratifiedShuffleSplit(y, 100, random_state=42)

    ### Setup the classifier using the provided model
    clf = Pipeline(steps=[("sel", SelectKBest(k=16)),
                          ("clf", model)])

    ### Create the custom scorer function
    custom_scorer = make_scorer(custom_score, greater_is_better=True)

    ### Create the grid and fit the classifier
    grid = GridSearchCV(clf, param_grid = param_grid, cv=cv, scoring=custom_scorer)
    grid.fit(X, y)

    ### Get the grid scores after fitted
    scores = grid.cv_results_['mean_test_score']
    scores = np.array(scores).reshape(len(n_estimator_range), len(learning_rate_range))

    ### Plot the heatmap
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=ddlheatmap)
    plt.xlabel('n_estimators')
    plt.ylabel('learning_rate')
    plt.colorbar()
    plt.xticks(np.arange(len(n_estimator_range)), n_estimator_range, rotation=45)
    plt.yticks(np.arange(len(learning_rate_range)), learning_rate_range)
    plt.title(
        "{} score of {:0.2f}.".format(grid.best_params_, grid.best_score_)
    )

    ### Store the figure
    plt.savefig('visual_grid_search.png')
    plt.close()

    print "Visual GridSearch created"
    print "..."

    return


def main():
    """main
    :rtype: None
    """
    print "main():: Load Dataset\n"

    ### Load the dictionary containing the dataset
    with open("final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)

    ### Remove outliers from dataset
    data_dict = remove_outliers(data_dict)

    ### Create a feature list
    features_list = create_features_list()

    ### Create new features
    features_list, data_dict = create_new_features(features_list, data_dict)

    ### Store to my_dataset for easy export below.
    my_dataset = data_dict

    ### Format the data, remove 'NaN' values
    formatted_data = featureFormat(my_dataset, features_list, sort_keys = True)

    ### Extract features and labels from dataset for local testing
    labels, features = targetFeatureSplit(formatted_data)

    ### Scale the features to standardize
    features = scale_features(features)

    ### Create a list of classifiers
    classifiers_list = create_classifiers_list()

    ### Test and score each classifier in the classifiers list
    test_classifiers(classifiers_list, my_dataset, features_list)

    ### Test and rank each feature in the features list
    test_features(AdaBoostClassifier(), features, labels, my_dataset, features_list)

    return

    ### Test the newly created features by calculating precision and recall without each new feature
    #REMOVE#test_new_features(AdaBoostClassifier(), features, labels, my_dataset, features_list)

    ### Tune the selected classifier to achieve better than .3 precision and recall
    params = tune_classifier(AdaBoostClassifier(), features, labels)
    visual_gridsearch(AdaBoostClassifier(), features, labels)

    ### Pipeline the MinMaxScaler since the test_classifier repartitions the non-scaled data
    clf = Pipeline(steps=[("sel", SelectKBest(k=params['sel__k'])),
                          ("clf", AdaBoostClassifier(n_estimators=params['clf__n_estimators'],
                                                     learning_rate=params['clf__learning_rate'],
                                                     algorithm=params['clf__algorithm']))])

    print "Running test_classifier() with final classifier parameters"

    ### Check the results using the test_classifier task
    test_classifier(clf, my_dataset, features_list)

    ### Task 6: Dump your classifier, dataset, and features_list so anyone can
    ### check your results. You do not need to change anything below, but make sure
    ### that the version of poi_id.py that you submit can be run on its own and
    ### generates the necessary .pkl files for validating your results.
    dump_classifier_and_data(clf, my_dataset, features_list)

    print "main() Completed"


if __name__ == '__main__':
    main()
