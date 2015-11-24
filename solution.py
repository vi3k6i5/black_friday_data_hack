import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from math import sqrt

# Read data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# categorical columns that i chose
categorical_columns = ["Product_ID", "Gender", "Age", "Occupation", "City_Category", "Stay_In_Current_City_Years",
                       "Marital_Status", "Product_Category_1", "Product_Category_2", "Product_Category_3"]

# label
train_y = np.array(train["Purchase"])

train_X = train.copy()
test_X = test.copy()

train_X = train_X.fillna(0)
test_X = test_X.fillna(0)

# I came up with a feature on what is the avg amount spent on a product id
# I tried a lot of other options here
# 1. Purchase price avg by gender, age group, product category 1, product category 2, product category 3
product_id_res = train_X.groupby(["Product_ID"])["Purchase"].mean()
avg_cost = train_X["Purchase"].mean()
# If i find a product id for which i dont have an avg pricing i will use global vg pricing.
product_id_res_map = {}
# created a map with product id to avg price map
val = product_id_res.iteritems()
for key, value in val:
    p_id = str(key)
    product_id_res_map[p_id] = value


def get_purchase_mean(product_id, product_category=None, key=None):
    key_pair = str(product_id)
    key_pair_pid = str(product_id) + str(product_category)
    if key == "1":
        if key_pair_pid in product_category_1_res:
            return product_category_1_res[key_pair_pid]
    elif key == "2":
        if key_pair_pid in product_category_2_res:
            return product_category_2_res[key_pair_pid]
    elif key == "3":
        if key_pair_pid in product_category_3_res:
            return product_category_3_res[key_pair_pid]
    if key_pair in product_id_res:
        return product_id_res[key_pair]
    return avg_cost

# Create a feature with pruduct_id to avg price of that product map
train_X["purchase_avg_by_p_id"] = map(lambda product_id: get_purchase_mean(product_id), train_X["Product_ID"])
test_X["purchase_avg_by_p_id"] = map(lambda product_id: get_purchase_mean(product_id), test_X["Product_ID"])

# Another feature that i created was
# Use_id to purchase power category
# Basically i came up with a distribution of purchase sum by suer.
# Created 10 hard coded buckets around it.

user_id_to_category_map = {}
customer_purchase_power = train_X.groupby("User_ID")["Purchase"].sum()
values = customer_purchase_power.iteritems()

for key, val in values:
    if val <= 146570.0:
        user_id_to_category_map[key] = 1
    elif val <= 205272.0:
        user_id_to_category_map[key] = 2
    elif val <= 279288.0:
        user_id_to_category_map[key] = 3
    elif val <= 383455.0:
        user_id_to_category_map[key] = 4
    elif val <= 521213.0:
        user_id_to_category_map[key] = 5
    elif val <= 698842.0:
        user_id_to_category_map[key] = 6
    elif val <= 942900.0:
        user_id_to_category_map[key] = 7
    elif val <= 1355245.0:
        user_id_to_category_map[key] = 8
    elif val <= 2069404.0:
        user_id_to_category_map[key] = 9
    else:
        user_id_to_category_map[key] = 10


def get_customer_category(user_id):
    if user_id in user_id_to_category_map:
        return user_id_to_category_map[user_id]
    return 5

# Tagged each user with a category id
train_X["user_category"] = map(lambda user_id: get_customer_category(user_id), train_X["User_ID"])
test_X["user_category"] = map(lambda user_id: get_customer_category(user_id), test_X["User_ID"])


# Encoding categorical variable with label encoding
for var in categorical_columns:
    lb = preprocessing.LabelEncoder()
    full_var_data = pd.concat((train_X[var], test_X[var]), axis=0).astype('str')
    lb.fit(full_var_data)
    train_X[var] = lb.transform(train_X[var].astype('str'))
    test_X[var] = lb.transform(test_X[var].astype('str'))

train_X = train_X.drop(['Purchase'], axis=1)

train_X = np.array(train_X)

# I built 3 models to make precictions
# Finally i did an avg of the 3 and submitted that.
print "1st model"
# 1st model
params = {}
params["objective"] = "reg:linear"
params["eta"] = 0.1
params["min_child_weight"] = 10
params["subsample"] = 0.7
params["colsample_bytree"] = 0.7
params["scale_pos_weight"] = 0.8
params["max_depth"] = 8
params["early_stopping_rounds"] = 10
params["seed"] = 42
plst = list(params.items())

xgtrain = xgb.DMatrix(train_X, label=train_y)
xgtest = xgb.DMatrix(test_X)
num_rounds = 1420

model = xgb.train(plst, xgtrain, num_rounds)

pred_test_y_xgb1 = model.predict(xgtest)

print "2nd model"
# 2nd model
# NOTE: I have changed the paramertes since i last uploaded the results. so the final score might vary.
params = {}
params["objective"] = "reg:linear"
params["eta"] = 0.1
params["min_child_weight"] = 10
params["subsample"] = 0.7
params["colsample_bytree"] = 0.7
params["scale_pos_weight"] = 0.8
params["max_depth"] = 8
params["early_stopping_rounds"] = 10
params["seed"] = 333
plst = list(params.items())

# This code shuffels the train matrix.
# In ensures that the oder of feature shuffel and label shuffel is same

merged_train_x_and_y = np.c_[train_X.reshape(len(train_X), -1), train_y.reshape(len(train_y), -1)]

shuffled_train_x = merged_train_x_and_y[:, :train_X.size//len(train_X)].reshape(train_X.shape)
shuffled_train_y = merged_train_x_and_y[:, train_X.size//len(train_X):].reshape(train_y.shape)

np.random.shuffle(merged_train_x_and_y)

# Shuffled train matrix is now shuffled_train_x
xgtrain = xgb.DMatrix(shuffled_train_x, label=shuffled_train_y)

model = xgb.train(plst, xgtrain, num_rounds)

pred_test_y_xgb2 = model.predict(xgtest)

print "3rd model"
# 3rd model
# NOTE: I have changed the paramertes since i last uploaded the results. so the final score might vary.
params = {}
params["objective"] = "reg:linear"
params["eta"] = 0.1
params["min_child_weight"] = 10
params["subsample"] = 0.7
params["colsample_bytree"] = 0.7
params["scale_pos_weight"] = 0.8
params["max_depth"] = 8
params["early_stopping_rounds"] = 10
params["seed"] = 777
plst = list(params.items())

# Shuffled train matrix again.
merged_train_x_and_y = np.c_[train_X.reshape(len(train_X), -1), train_y.reshape(len(train_y), -1)]

shuffled_train_x = merged_train_x_and_y[:, :train_X.size//len(train_X)].reshape(train_X.shape)
shuffled_train_y = merged_train_x_and_y[:, train_X.size//len(train_X):].reshape(train_y.shape)

np.random.shuffle(merged_train_x_and_y)

xgtrain = xgb.DMatrix(shuffled_train_x, label=shuffled_train_y)

model = xgb.train(plst, xgtrain, num_rounds)

pred_test_y_xgb3 = model.predict(xgtest)

test['Purchase'] = (pred_test_y_xgb1 + pred_test_y_xgb2 + pred_test_y_xgb3) / 3
test.to_csv('final_xgb.csv', columns=['User_ID', 'Product_ID', 'Purchase'], index=False)
