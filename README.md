# Web Phishing Detection
The main goal of this repository is to identify and classify phishing websites using deep learning techniques. Phishing websites are malicious sites that try to trick users into providing sensitive information, such as usernames, passwords, and credit card details, by pretending to be trustworthy entities. The raw data being analyzed here are URLs and HTML content of websites. URL analysis involves examining the structure and contents of the URL for signs of phishing, such as misspelled domain names or suspicious subdomains. HTML content analysis involves examining the actual contents of the webpage, such as forms that ask for sensitive information, the presence of suspicious scripts, or the overall structure and design of the page.



# Dataset
The 'look-before-you-leap' dataset, accessible on Kaggle, is employed in this project. It's a balanced dataset consisting of 45,373 instances, equally representing both benign and phishing web pages. Each instance encompasses a variety of HTML document elements such as texts, hyperlinks, images, tables, lists, and diverse URL components from subdomains to queries.

The dataset's creator has already prudently removed URL prefixes like HTTP:// and HTTPS:// from the dataset. This essential modification allows the model to focus on the more critical parts of the URL. It also guarantees the model's consistent performance across different URL datasets, enhancing its generality and avoiding skewed results.

The dataset comprises real-world data collected from Alexa.com for legitimate web pages and phishtank.com for phishing web pages. The use of these trusted sources guarantees a realistic data mix, creating a robust and authentic training environment for the deep learning model.


# Exploratory Data Analysis


URL Dataset Shape: (45373, 4)
HTML Dataset Shape: (45373, 4)

Total URL spam:  22686
Total URL ham:  22686
Total rows: 45373

Total HTML spam:  22686
Total HTML ham:  22687
Total rows: 45373

Missing Values on URL:
Category    0
Data        0
dtype: int64

Missing Values on HTML:
Category    0
Data        0
dtype: int64


# Data Visualization

### URL and HTML Distribution
The distribution of HTML and URL is visualized using donut charts.

![image](https://github.com/aridofflimits/Web-Phishing-Detection/assets/147245715/d485510d-6a03-4e18-b9ec-282a0d557b3e)

### Distribution of Categories
The category distribution in both datasets is visualized using bar charts, which helps in understanding class balance and may inform the need for stratification or rebalancing techniques.

![image](https://github.com/aridofflimits/Web-Phishing-Detection/assets/147245715/b34eb679-830b-4b70-b608-7c9770c9d79c)

![image](https://github.com/aridofflimits/Web-Phishing-Detection/assets/147245715/5e4ac6e6-c94c-4796-814f-3a7dbfb9ae09)

### Distribution Percentages
The percentage distribution of categories in both datasets is calculated and printed to provide a clearer view of class imbalance in percentage terms.

### Visualizing Category Distribution Percentages
Donut charts are created to visually represent the percentage distribution of categories in both datasets, offering an intuitive understanding of class proportions.

![image](https://github.com/aridofflimits/Web-Phishing-Detection/assets/147245715/1fbbebed-56dc-4feb-b22a-486e505b0e0e)

![image](https://github.com/aridofflimits/Web-Phishing-Detection/assets/147245715/636dd305-9d46-4511-808d-4488098b4065)

### URL and HTML Length Analysis
The length of URLs and HTML content with respect to their categories is analyzed and visualized using box plots. This can reveal patterns such as longer or shorter lengths being associated with phishing/spam or legitimate content, which might be useful features for modeling.

![image](https://github.com/aridofflimits/Web-Phishing-Detection/assets/147245715/7deb2ef0-7c54-4cee-a1d7-367de4fc056d)

![image](https://github.com/aridofflimits/Web-Phishing-Detection/assets/147245715/538050c4-e68a-4122-9c07-4eec102e0c1c)


### Model Architecture

```
Model: "model_1"
__________________________________________________________________________________________________
 Layer (type)                Output Shape                 Param #   Connected to                  
==================================================================================================
 url_input (InputLayer)      [(None, 180)]                0         []                            
                                                                                                  
 html_input (InputLayer)     [(None, 2000)]               0         []                            
                                                                                                  
 url_embedding (Embedding)   (None, 180, 16)              608       ['url_input[0][0]']           
                                                                                                  
 html_embedding (Embedding)  (None, 2000, 16)             160000    ['html_input[0][0]']          
                                                                                                  
 conv1d_2 (Conv1D)           (None, 171, 64)              10304     ['url_embedding[0][0]']       
                                                                                                  
 conv1d_3 (Conv1D)           (None, 1991, 64)             10304     ['html_embedding[0][0]']      
                                                                                                  
 max_pooling1d_2 (MaxPoolin  (None, 85, 64)               0         ['conv1d_2[0][0]']            
 g1D)                                                                                             
                                                                                                  
 max_pooling1d_3 (MaxPoolin  (None, 995, 64)              0         ['conv1d_3[0][0]']            
 g1D)                                                                                             
                                                                                                  
 flatten_2 (Flatten)         (None, 5440)                 0         ['max_pooling1d_2[0][0]']     
                                                                                                  
 flatten_3 (Flatten)         (None, 63680)                0         ['max_pooling1d_3[0][0]']     
                                                                                                  
 concatenate_layer (Concate  (None, 69120)                0         ['flatten_2[0][0]',           
 nate)                                                               'flatten_3[0][0]']           
                                                                                                  
 dense1 (Dense)              (None, 64)                   4423744   ['concatenate_layer[0][0]']   
                                                                                                  
 dense2 (Dense)              (None, 32)                   2080      ['dense1[0][0]']              
                                                                                                  
 output_layer (Dense)        (None, 1)                    33        ['dense2[0][0]']              
                                                                                                  
==================================================================================================
Total params: 4607073 (17.57 MB)
Trainable params: 4607073 (17.57 MB)
Non-trainable params: 0 (0.00 Byte)

```


# Training Model

```

### Early stopping to prevent overfitting early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
### Model Training history = model.fit( [url_X_train, html_X_train], # URL and HTML training data url_y_train, validation_data=([url_X_test, html_X_test], url_y_test), # URL and HTML validation data epochs=30, # Number of epochs to train for batch_size=8, # Batch size for training callbacks=[early_stopping] # Early stopping callback )
Epoch 1/30
4538/4538 [==============================] - 287s 62ms/step - loss: 0.1446 - accuracy: 0.9446 - precision_1: 0.9460 - recall_1: 0.9432 - val_loss: 0.0876 - val_accuracy: 0.9699 - val_precision_1: 0.9711 - val_recall_1: 0.9683
Epoch 2/30
4538/4538 [==============================] - 71s 16ms/step - loss: 0.0501 - accuracy: 0.9829 - precision_1: 0.9837 - recall_1: 0.9821 - val_loss: 0.0914 - val_accuracy: 0.9684 - val_precision_1: 0.9507 - val_recall_1: 0.9876
Epoch 3/30
4538/4538 [==============================] - 58s 13ms/step - loss: 0.0247 - accuracy: 0.9918 - precision_1: 0.9922 - recall_1: 0.9914 - val_loss: 0.0952 - val_accuracy: 0.9732 - val_precision_1: 0.9783 - val_recall_1: 0.9676
Epoch 4/30
4538/4538 [==============================] - 50s 11ms/step - loss: 0.0146 - accuracy: 0.9956 - precision_1: 0.9952 - recall_1: 0.9961 - val_loss: 0.1209 - val_accuracy: 0.9736 - val_precision_1: 0.9747 - val_recall_1: 0.9721

```


# Evaluation Model

```
y_pred_prob = model.predict([url_X_test, html_X_test]) y_pred = (y_pred_prob > 0.5).astype(int)
284/284 [==============================] - 1s 3ms/step
# Calculating metrics accuracy = accuracy_score(url_y_test, y_pred) precision = precision_score(url_y_test, y_pred) recall = recall_score(url_y_test, y_pred) f1 = f1_score(url_y_test, y_pred)
# Printing evaluation metrics print(f"Accuracy: {accuracy:.4f}") print(f"Precision: {precision:.4f}") print(f"Recall: {recall:.4f}") print(f"F1 Score: {f1:.4f}")
Accuracy: 0.9699
Precision: 0.9711
Recall: 0.9683
F1 Score: 0.9697


```





