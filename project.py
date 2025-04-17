##------ IMPORTS  
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix


st.title("Predicting a Deal's Success")

"Cami Krugel - 000828616"

tab1, tab2 = st.tabs(["Report", "Try it Out!"])

with tab1:
    st.title("Introduction")
    st.subheader("Problem")
    "Right now, a company has an abundence of data about their potential deals and past deals. There is enough information about deals that have succeeded and those that have not to be able to draw insights about the factors that contribute. However, the data is not being used to help the company as it could. If the company can predict whether or not a deal will be a success, it could better allocate their resources to prioritize the deals that may be more successful. "
    st.subheader("Motivation")
    "I want to work on the Sales Success Playbook Project as I am excited about getting to work on a real problem with real-world data. Also, I have the oppertunity to impact an organization and help them better allocate their resources and predict their success."
    st.subheader("Objectives")
    "I am excited to pull from the deal and company datasets to create a dashboard that employees can use to see my models predictions about their deals based on their input. I hope this can help sales representatives make better decisions based on the predictions."


    st.title("Data Description & Preprocessing")

    #import data and mappings
    companies = pd.read_csv('/mount/src/salesplaybook/anonymized_hubspot_companies.csv')
    deals =  pd.read_csv('/mount/src/salesplaybook/anonymized_hubspot_deals.csv')
    with open('/mount/src/salesplaybook//mappings.json', 'r') as f:
        mappings = json.load(f)

    st.subheader("Source")
    "This data came from SymTrain's sales playbook, where they have a a variety of company and deal data. The target variable I will be trying to predict is whether or not the deal was won."

    with st.echo():
        deal_to_company = {}

        for company_id, deal_list in mappings["CompanyToDeals"].items():
            for deal_id in deal_list:
                deal_to_company[deal_id] = company_id

        #### create a company for company id based on the dictionary using mapping function
        deals['Record ID'] = deals['Record ID'].astype(str)
        deals['Company ID'] = deals['Record ID'].map(deal_to_company)
        companies['Record ID'] = companies['Record ID'].astype(str)

        # merge the company and deal deatasets together based on their IDs
        deals_merged = pd.merge(left = deals, right = companies, left_on ='Company ID'  ,right_on ='Record ID' , how = 'left')

        # keep only closed deals for training
        deals_merged = deals_merged[deals_merged['Is Deal Closed?']]


    st.subheader("Data Cleaning")
    "First, I had to remove the irrelevant or duplicate columns from the data"

    with st.echo():
        # keep relevant columns
        deals_merged = deals_merged[[
            'Deal Score', 'Deal source attribution 2', 'Original Traffic Source', 'Annual recurring revenue', 'Monthly recurring revenue', 'Is Closed Won', 'Forecast probability','Contract End Date_x',  'Contract Term (Months)',
            'Trial Start date', 'Trial End Date', 'Amount', 'Total contract value', 'Deal Type',  'Parent Company', 'CCaaS', 'Annual Revenue', 'Target Account',  'Number of Form Submissions', 'Total Agents',  '# of Agents Total', 
            'Number of times contacted', 'Revenue range','# of Agents Contracted', 'Time Zone', 'Number of Pageviews', 'Primary Company','Year Founded', 'ICP', 'Industry group', 'Segmentation', 'LMS System','SymTrain Use Cases',  
            'BPO', 'SSO Application','SymTrain Product', 'SSO Implemented?', 'Consolidated Industry', 'Number of Employees', 'BPO Program', 'Number of Sessions', 'WFM', 'Country/Region','BPO Program Tier']]


    st.subheader("Handling Missing Values")
    "Next, I removed or filled missing values from the data."
   
    with st.echo():
        
        deals_full = deals_merged.loc[:,deals_merged.isnull().sum() < len(deals_merged)/2]
        # keep data that is only missing less than half of the values

        for col in deals_full:
        # fill remaining missing values
                if not is_numeric_dtype(deals_full[col]):
                        # for categorical variables
                        if max(deals_full[col].value_counts(normalize = True, dropna = False))>.8:
                                # if median has majority, fill na with that
                                deals_full[col]= deals_full[col].fillna(value = deals_full[col].value_counts(normalize = True, dropna = False, sort = True).index[0])
                        else:
                                # if there is no large majority, drop na values
                                deals_full= deals_full.dropna(subset=col, axis=0)
                        # for each unqiue value in the column...
                        for value, pct in deals_full[col].value_counts(normalize = True, dropna = False).items():
                                # if less than 10% of the data has that column, put it in the "other" category 
                                if pct < .1:
                                        deals_full[col]=deals_full[col].replace(value, 'Other')

                else:
                        # numerical variables
                        if abs((deals_full[col].mean() - deals_full[col].median())/deals_full[col].std()) > .5:
                                # if the data is skewed, fill na with the median values
                                deals_full[col]= deals_full[col].fillna(value = deals_full[col].median())
                        else:
                                # if the data is normally distributed, fill na with the means
                                deals_full[col]= deals_full[col].fillna( value = deals_full[col].mean())

    st.subheader("Transformations")
    "For the remaining categorical variables, since they are all nominal, fill with one-hot encoding."

    with st.echo():
            deals_dummies = pd.get_dummies(deals_full, columns = ['Deal source attribution 2', 'Original Traffic Source',  'Deal Type',  'Country/Region', 
                                                            'Time Zone','Consolidated Industry'], dtype = int)
    st.subheader("Feature Selection")
    "The data still has a lot of variables. I have determined that removing variabels with less that 5% correlation to the target variable will help reduce complexity without comprimising accuracy. Also, I can delete the \"Other\" categories, as they are the same as a zero in the related ones."
    
    with st.echo():    
        deals_dummies = deals_dummies.loc[:,abs(deals_dummies.corr(method='pearson')["Is Closed Won"])>.05]
        # drop features with less than 5% correlation
        deals_dummies=deals_dummies.drop(['Time Zone_America/Chicago', 'Time Zone_America/Los_Angeles', 'Time Zone_Other', 'Original Traffic Source_Other','Country/Region_Other','Deal Type_Other', 'Consolidated Industry_Other' ], axis =1)
        # drop "other" features

    #sns.pairplot(deals_dummies, hue = "Is Closed Won" ,palette='coolwarm')
    st.image('/mount/src/salesplaybook/pairplot.png')
    "Evaluating the pairplot, the data doesn't look like it has features or relationships that are significantly less impactful than others for the target variable, so we will continue with these features. Now, we can remove the target variable to create our X and y data."
    
    with st.echo():     
        X= deals_dummies.drop("Is Closed Won", axis =1)
        y= deals_dummies.loc[:,"Is Closed Won"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    st.subheader("Feature Scaling")
    "Since we will be using methods that are sensitive to scale, we normalize it before going further."
    with st.echo():
        mean = np.mean(X_train, axis = 0)
        std = np.std(X_train, axis = 0)
        X_train_norm = (X_train-mean)/std
        X_test_norm = (X_test-mean)/std

    st.subheader("PCA")
    "The data still has 13 components, so we will use Principal Component Analysis to reduce the factors."

    with st.echo():
        pca =  PCA(n_components = 13)
        pca.fit(X_train_norm)
    
    plt.plot(np.cumsum(pca.explained_variance_ratio_),'k*-')
    plt.grid()
    plt.xlabel('Number of principal components')
    plt.ylabel('Acucmulative variance')
    st.pyplot(plt)
    "In visualizing the explained varience, we see that after 9 components, the model will not perform enough better to justify the increase in complexity."

    with st.echo():
        pca =  PCA(n_components = 9)
        pca.fit(X_train_norm)
       
       # transform both the training and test data
        train_components = pca.transform(X_train_norm) 
        test_components = pca.transform(X_test_norm) 

    st.title("Methodology")
   
    st.subheader("Logistic Regression")
    "Logistic regression has no hyperparameters, so we can create the model as-is."
    
    with st.echo():
        lgr_model = LogisticRegression(solver='sag', max_iter=100000, penalty=None)
        lgr_model.fit(train_components,y_train) 

    st.subheader("KNN")
    "KNN has the hyperparameter of k(number of neighbors). So, we will test multiple values and run cross validation to see the best value for the output."

    scores_list = []

    with st.echo():
        num_neighbors_list = [1,3,5,25,50]
         
         # try all K values
        for k in num_neighbors_list:
            # define the KNN model and pass in the K
            knn = KNeighborsClassifier(n_neighbors=k)
            # obtain scores with 5 folds
            scores_knn = cross_val_score(knn, train_components, y_train, cv = 5)
            # print the average accuracy 
            st.write(k, "- ", np.mean(scores_knn))
            scores_list.append(scores_knn)
            
    "We see the optimal value is 25, so we can set k to 25."
    with st.echo():
         knn = KNeighborsClassifier(n_neighbors=25)
         knn.fit(train_components, y_train)


    "a. Description and explanation of the chosen machine learning algorithms"
    "b. Justification for model selection"
    "d. Baseline methods for comparison (if applicable)"

    st.title("Results & Evalution")
    st.subheader("Model Performance Metrics")

    models  = ["Logistic", "KNN"]
    y_pred_l= lgr_model.predict(test_components)
    y_pred_k = knn.predict(test_components)
    y_pred = {"Logistic": y_pred_l, "KNN":y_pred_k}
    y_probs_k = knn.predict_proba(test_components)[:,1]
    y_probs_l = lgr_model.predict_proba(test_components)[:,1]
    y_probs = {"Logistic": y_probs_l, "KNN":y_probs_k}

    accuracy = dict()
    fpr = dict()
    tpr = dict()
    auc_tem = dict()
    f1 = dict()
    for model in models:
        accuracy[model] = sum(y_test == y_pred[model])/len(y_test)
        fpr[model], tpr[model], thresholds = roc_curve(y_test, y_probs[model])
        auc_tem[model] = auc(fpr[model], tpr[model])
        f1[model] = f1_score(y_test, y_pred[model])
    dict2 = dict()
    dict2 ["Logistic"] = accuracy["Logistic"], auc_tem["Logistic"], f1["Logistic"]
    dict2 ["KNN"] = accuracy["KNN"], auc_tem["KNN"], f1["KNN"]

    results = pd.DataFrame(dict2,index=["Accuracy", "AUC", "F1"])
    st.table(results)

    plt.figure()
    for model in models:
        plt.plot(
            fpr[model],
            tpr[model],
            label= model % auc_tem[model],
        )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    plt.show()
    st.pyplot(plt)


    col1, col2 = st.columns([1, 1])
    with col1:
        cm_l  = confusion_matrix(y_test, y_pred_l)
        plt.figure(figsize=(5, 4))
        plt.title("Logistic Confusion Matrix")
        sns.heatmap(cm_l, annot=True, fmt="d", cmap="Blues", xticklabels=["Predicted Loss", "Predicted Win"], yticklabels=["Actual Loss", "Actual Win"])
        st.pyplot(plt)
    with col2:
        cm_k  = confusion_matrix(y_test, y_pred_k)
        plt.figure(figsize=(5, 4))
        plt.title("KNN Confusion Matrix")
        sns.heatmap(cm_k, annot=True, fmt="d", cmap="Blues", xticklabels=["Predicted Loss", "Predicted Win"], yticklabels=["Actual Loss", "Actual Win"])
        st.pyplot(plt)

#For classification tasks, evaluate performance using at least accuracy, AUC, and F1 score.

    #print("KNN mean:\n",np.mean(scores_list[3]),"\nKNN std:\n",np.std(scores_list[3]))

   # print("Linear regression mean:\n",np.mean(scores_lr),"\nLinear Regression std:\n",np.std(scores_lr))
    
    st.subheader("Comparison")
    "Discussion of strengths and limitations"

    st.title("Conclusion & Future Work")
    st.subheader("Key Findings")
    st.subheader("Next Steps")

    st.title("References")


    st.title("Appendix")


with tab2:
    st.title("Predict a New Deal")

    form =  st.slider('Number of Form Submissions', max_value = int(max(deals_merged['Number of Form Submissions'])))
    pv =  st.slider('Number of Pageviews', max_value = int(max(deals_merged['Number of Pageviews'])))
    emp =  st.slider('Number of Employees', max_value = int(max(deals_merged['Number of Employees'])))
    ses =  st.slider('Number of Sessions', max_value = int(max(deals_merged['Number of Sessions'])))
    dsa = st.selectbox(
        'Deal source attribution',
        np.array(deals_merged['Deal source attribution 2'].dropna().unique())
    )
    ts = st.selectbox(
        'Original Traffic Source',
        np.array(deals_merged['Original Traffic Source'].dropna().unique()))
    country = st.checkbox('From the USA?')
    dt = st.selectbox(
        'Deal Type',
        np.array(deals_merged['Deal Type'].dropna().unique()))
    ci = st.selectbox(
        'Consolidated Industry',
        np.array(deals_merged['Consolidated Industry'].dropna().unique()))

    df = pd.DataFrame()

    df= {'Number of Form Submissions' : form,
    'Number of Pageviews' : pv, 
    'Number of Employees' : emp,
    'Number of Sessions' : ses,
    'Deal source attribution 2_Event' : 1 if dsa =='Event' else 0,
    'Deal source attribution 2_Referral Partner' : 1 if dsa =='Rerferral Partner' else 0,
    'Original Traffic Source_Offline Sources' : 1 if ts =='Offline Sources' else 0,
    'Deal Type_New' : 1 if dsa =='New' else 0,
    'Deal Type_Renewal' : 1 if dsa =='Renewal' else 0,
    'Country/Region_United States' : 1 if country else 0,
    'Consolidated Industry_BPO' : 1 if ci =='BPO' else 0,
    'Consolidated Industry_Banking' : 1 if ci =='Banking' else 0,
    'Consolidated Industry_Healthcare' : 1 if ci =='Healthcare' else 0}

    new = pd.DataFrame([df]) 
    vals = new.iloc[0,:]
    vals = np.array((vals-mean)/std).reshape(1, -1)

    new = pca.transform(vals)

    col1, col2 = st.columns([2, 1])


    with col1:
        st.subheader("Logistic Prediction")
        'The deal will be a success' if lgr_model.predict(new) else "The deal won't be a success"

    with col2:
        st.subheader("KNN Prediction")
        'The deal will be a success' if knn.predict(new) else "The deal won't be a success"

