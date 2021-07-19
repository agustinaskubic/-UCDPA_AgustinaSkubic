# Importing Files and Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

ffc = pd.read_csv('/Users/aguiskubic/Desktop/UCD Final Project/archive/fulfilment_center_info.csv')
meal = pd.read_csv('/Users/aguiskubic/Desktop/UCD Final Project/archive/meal_info.csv')
df = pd.read_csv('/Users/aguiskubic/Desktop/UCD Final Project/archive/train.csv')
sample = pd.read_csv('/Users/aguiskubic/Desktop/UCD Final Project/archive/sample_submission.csv')

# Information from the tables

# Information from df table

print('------------ DF INFORMATION ------------')
print(np.shape(df))
print(df.info())
print(df.head())
print(
    len(df.id.unique()))  # that's how we know that id column is the primary key since there is one unique id for each row

# Information from ffc table

print('------------ FFC INFORMATION ------------')
print(np.shape(ffc))
print(ffc.info())
print(ffc.head())
print(len(ffc.center_id.unique()))  # that's how we know that center_id column is the primary key

# Count the unique values in each column (we count for the first three columns since they are just id and codes)
center_count = pd.unique(ffc.center_id)
city_count = pd.unique(ffc.city_code)
region_count = pd.unique(ffc.region_code)

# initializing variable to 0 for counting
center_unique = 0
city_unique = 0
region_unique = 0

# writing separate for loop of each groups
for x in center_count:
    center_unique += 1

for x in city_count:
    city_unique += 1

for x in region_count:
    region_unique += 1

print(f'Count of center_id : {center_unique}')
print(f'Count of city_code : {city_unique}')
print(f'Count of region_code : {region_unique}')
print(ffc.center_type.unique())  # to see the different types of centers
print(ffc.op_area.unique())

# Information from meal table

print('------------ MEAL INFORMATION ------------')
print(np.shape(meal))
print(meal.info())
print(meal.head())
print(len(meal.meal_id.unique()))  # that's how we know that meal_id column is the primary key
list_cat = pd.unique(meal['category'])
list_cui = pd.unique(meal['cuisine'])
print(list_cat)  # to see the different categories
print(list_cui)  # to see the different cuisines

# Joining tables

df_ffc = df.merge(ffc, on= 'center_id', how = 'inner')
df_meal = df.merge(meal, on = 'meal_id',how = 'inner')
df_meal_ffc = df.merge(meal, on = 'meal_id') \
                    .merge(ffc, on='center_id')

print(df_ffc.head())
print(np.shape(df_ffc))
print(df_meal.head())
print(np.shape(df_meal))
print(df_meal_ffc.head())
print(np.shape(df_meal_ffc))

# Cleaning the data (double checking since in the description I have noticed no null values)

#Checking missing values
missing_values_ffc = ffc.isnull().sum().values
print('The fulfilment_center dataset has NO null values') if np.sum(missing_values_ffc) == 0 else print('The fulfilment_center dataset has null values')

missing_values_df = df.isnull().sum().values
print('The train dataset has NO null values') if np.sum(missing_values_df) == 0 else print('The train dataset has null values')

missing_values_meal = meal.isnull().sum().values
print('The meal_info dataset has NO null values') if np.sum(missing_values_df) == 0 else print('The meal_info dataset has null values')

#Checking n/a values

na_values_ffc = ffc.isna().sum()
na_ffc = na_values_ffc.to_numpy()
print('The fulfilment_center dataset has NO na values') if np.sum(na_ffc) == 0 else print('The fulfilment_center dataset has null values')

na_values_df = df.isna().sum()
na_df = na_values_df.to_numpy()
print('The train dataset has NO na values') if np.sum(na_df) == 0 else print('The train dataset has null values')

na_values_meal = meal.isna().sum()
na_meal = na_values_meal.to_numpy()
print('The meal_info dataset has NO na values') if np.sum(na_meal) == 0 else print('The meal_info dataset has null values')

# There are non null values in the dataset. However the following code should be executed if we want to replace them with 0 to not lose any data:

cleaned_data1 = ffc.fillna(0)
cleaned_data2 = df.fillna(0)
cleaned_data3 = meal.fillna(0)

# Duplicated values

drop_duplicates= df.drop_duplicates()
print('The train dataset has NO duplicated values') if df.shape == drop_duplicates.shape else print('The train dataset has duplicated values')

# Anayzing the data: ffc & meal

# center_id group by Center_type

center_id_type = ffc.groupby('center_type').agg({'center_id':'count'}).sort_values(['center_type', 'center_id'], ascending = [True, False])
print(center_id_type)

center_id_type.plot(kind='bar', color = ['purple'], rot = 0)
plt.title('Count of center_id by Type')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel("Center Type")
plt.ylabel("Count of center_id")
plt.show()

# center_id group by region_code

center_id_region = ffc.groupby('region_code').agg({'center_id':'count'}).sort_values(['center_id', 'region_code'], ascending = [False, True])
print(center_id_region)

center_id_region.plot(kind='bar', color = ['purple'], rot = 0)
plt.title('Count of center_id by Region')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel("Regions")
plt.ylabel("Count of center_id")
plt.show()

# meal_id group by category

meal_category = meal.groupby('category').agg({'meal_id': 'count'}).sort_values(['meal_id', 'category'], ascending = [False, True])
print(meal_category)

meal_category.plot(kind='bar', color = ['goldenrod'], rot = 45)
plt.title('Count of meal_id by Category')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel("Categories")
plt.ylabel("Count of meal_id")
plt.show()

# meal_id group by cuisine

meal_cuisine = meal.groupby('cuisine').agg({'meal_id': 'count'}).sort_values(['meal_id', 'cuisine'], ascending = [False, True])
print(meal_cuisine)

meal_cuisine.plot(kind='bar', color = ['goldenrod'], rot = 0)
plt.title('Count of meal_id by Cuisine')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel("Cuisines")
plt.ylabel("Count of meal_id")
plt.show()

# Analysing the data: df_ffc

# Orders by region - General info

orders_region_code = df_ffc.groupby('region_code')['num_orders'].agg([min, max, np.mean, np.median]).sort_values('max', ascending = False)
price_region_code = df_ffc.groupby('region_code')['checkout_price'].agg([min, max, np.mean, np.median]).sort_values('max', ascending = False)

print(orders_region_code)
print(price_region_code)

# number of orders by region & checkout_price

df_ffc['region_code'] = df_ffc['region_code'].astype(str)

sns.set(style='whitegrid')
sns.scatterplot(x="checkout_price",
                    y="num_orders",
                    hue="region_code",
                    data=df_ffc,
                palette= 'plasma',
               marker = '+')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel("Checkout price")
plt.ylabel("Number of orrders")
plt.title('Number of orders and Checkout price by Region')

plt.show()

# total orders by region in percentage

total_orders_region = df_ffc[['region_code', 'num_orders']].groupby('region_code', as_index = False).agg({'num_orders':'sum'}).sort_values('num_orders', ascending = False)

total_orders_region['percent'] = (total_orders_region['num_orders'] / total_orders_region['num_orders'].sum()) * 100

print(total_orders_region)

# number of orders over the weeks

orders_week = df_ffc[['region_code', 'week', 'num_orders']].groupby(['region_code', 'week'], as_index = False).agg({'num_orders':'sum'}).sort_values('num_orders', ascending = False)

orders_week['region_code'] = orders_week['region_code'].astype(str) #change the data type in order to be readeble on the chart

sns.lineplot(x="week", y="num_orders", data=orders_week, hue="region_code", palette = 'plasma')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel("Week")
plt.ylabel("Number of orders")
plt.title('Number of orders by Region through time')
plt.show()

# Anayzing the data: df_meal

# orders by categories - General info

orders_category = df_meal.groupby('category')['num_orders'].agg([min, max, np.mean, np.median]).sort_values('max', ascending = False)

print(orders_category)

# total orders by categories

total_orders_cat = df_meal[['category', 'num_orders']].groupby('category', as_index = False).agg({'num_orders':'sum'}).sort_values('num_orders', ascending = False)

sns.barplot(x ='category', y ='num_orders', data = total_orders_cat,
            color ='darkmagenta').set_title('Number of orders by Category')
plt.xlabel("Categories", fontsize=12)
plt.xticks(rotation=45)
plt.ylabel("Number of orders", fontsize=12)
plt.show()

# orders by cuisine - General info

orders_cuisine = df_meal.groupby('cuisine')['num_orders'].agg([min, max, np.mean, np.median]).sort_values('max', ascending = False)

print(orders_cuisine)

# total orders by cuisine

total_orders_cui = df_meal[['cuisine', 'num_orders']].groupby('cuisine', as_index = False).agg({'num_orders':'sum'}).sort_values('num_orders', ascending = False)

sns.barplot(x ='cuisine', y ='num_orders', data = total_orders_cui,
            color ='goldenrod').set_title('Number of orders by Cuisine in millions')
plt.xlabel("Cuisines", fontsize=12)
plt.ylabel("Number of orders", fontsize=12)
plt.show()

#  most popular category by couisine

cat_cui = df_meal[['cuisine', 'category', 'num_orders']].groupby(['cuisine', 'category']).agg({'num_orders':'sum'}).sort_values(['cuisine','num_orders'], ascending = [True, False])

print(cat_cui)

df_meal_1 = df_meal[df_meal['num_orders'] < 2000]

sns.violinplot(x="cuisine", y="num_orders", data=df_meal_1, palette="plasma", bw=.2, cut=1)
plt.xlabel("Cuisines")
plt.ylabel("Number of orders")
plt.title('Distribution of orders by Cuisine')
plt.show()

# number of orders & checkout price by cuisine

sns.set(style='whitegrid')
sns.scatterplot(x="num_orders",
                    y="checkout_price",
                    hue="cuisine",
                    data=df_meal,
                palette= 'plasma')
plt.xlabel("Number of orders")
plt.ylabel("Checkout price")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('Number of orders and Checkout price by Cuisine')

plt.show()

# total orders of the most popular cuisine (Italian)

orders_italy = df_meal[df_meal['cuisine'] == 'Italian'].agg({'num_orders':'sum'})

print(orders_italy)

# Analysing the prices

# base price vs checkout price

sns.jointplot(
    data = df_meal_ffc,
    x='base_price', y='checkout_price', hue='cuisine',
    kind="kde", palette = 'plasma'
)
plt.show()

# price by cuisine -- General info

price_cuisine = df_meal.groupby('cuisine')['checkout_price'].agg([min, max, np.mean, np.median]).sort_values('max', ascending = False)

print(price_cuisine)

# checkout_price by meal_id

df1 = df[['meal_id','checkout_price']].drop_duplicates(subset='checkout_price').groupby('meal_id').agg({'checkout_price':'count'}).sort_values('checkout_price', ascending = False)

print(df1.head()) # it can be seen that there are different checkout prices by meal_id. This can be for many reasons such us: discount, different prices by region, etc.

sns.violinplot(x="cuisine", y="checkout_price", data=df_meal_ffc, palette="plasma", bw=.2, cut=1, linewidth=1)
plt.xlabel("Cuisines")
plt.ylabel("Checkout price")
plt.title('Checkout price by Cuisine')
plt.show()

# average price per cuisine

df_meal.boxplot(by ='cuisine', column =['checkout_price'], color = 'purple')
plt.suptitle('')
plt.xlabel("Cuisines")
plt.show()

# categories & prices -- GENERAL INFO

category_price = df_meal.groupby('category')['checkout_price'].agg([min, max, np.mean, np.median]).sort_values('max', ascending = False)

print(category_price)


figure(figsize = (12,6), dpi=80)
sns.violinplot(x="category", y="checkout_price", data=df_meal_ffc, palette="plasma", bw=.2, cut=1, linewidth=1)
plt.xticks(rotation=50)
plt.xlabel("Categories")
plt.ylabel("Checkout price")
plt.title('Checkout price by Cuisine and Category')
plt.show()

# Analysing the data: df_meal_ffc

# see if the meal _id was sold with discount

def discount(data):
    return 1 if (data['base_price'] - data['checkout_price'])> 0 else 0

df_meal_ffc["discount"] = df_meal_ffc.apply(lambda x: discount(x), axis=1)

#print(df_meal_ffc.head())

def total(data):
    return 1

df_meal_ffc["total"] = df_meal_ffc.apply(lambda x: total(x), axis=1)

# average weighted of discounts by region

region_discount = df_meal_ffc[['region_code', 'discount']].groupby('region_code', as_index = False).agg({'discount':'sum'}).sort_values('discount', ascending = False)

region_total = df_meal_ffc[['region_code', 'total']].groupby('region_code', as_index = False).agg({'total':'sum'}).sort_values('total', ascending = False)

region_discount['percent'] = region_discount['discount'] / region_total['total']

print(region_discount)

sns.barplot(x ='region_code', y ='percent', data = region_discount,
            color ='darkmagenta').set_title('Percetange of discount by Region')
plt.xlabel("Regions")
plt.ylabel("Percentage of discount")
plt.show()

# average weighted of discounts by category

cat_discount = df_meal_ffc[['category', 'discount']].groupby('category').agg({'discount':'sum'}).sort_values('discount', ascending = False)

cat_total = df_meal_ffc[['category', 'total']].groupby('category').agg({'total':'sum'}).sort_values('total', ascending = False)

cat_discount['percent'] = cat_discount['discount'] / cat_total['total']

print(cat_discount)

# discounts by category and region

discount_cat_region = df_meal_ffc[['region_code','category','discount']].groupby(['region_code','category']).agg({'discount':'sum'}).sort_values(['discount', 'category'], ascending = [False, False])
print(discount_cat_region)

# discounts by cuisine and region

discount_cui_region = df_meal_ffc[['region_code','cuisine','discount']].groupby(['region_code','cuisine'], as_index = False).agg({'discount':'sum'}).sort_values(['region_code', 'discount'], ascending = [False, False])
print(discount_cui_region)

sns.barplot(x ='region_code', y ='discount',hue = 'cuisine', data = discount_cui_region,
            palette ='plasma').set_title('Count of discount by Cuisine by Region')
plt.xlabel("Regions", fontsize=12)
plt.xticks(rotation=45)
plt.ylabel("Discount", fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

# emailer_for_promotion

# mean price for those who have email promo

email_promo = df_meal_ffc[df_meal_ffc['emailer_for_promotion']==1].groupby(['category']).agg({'checkout_price':'mean'}).sort_values('checkout_price', ascending = False)

# mean price for those who have not email promo

email_no_promo = df_meal_ffc[df_meal_ffc['emailer_for_promotion']!=1].groupby(['category']).agg({'checkout_price':'mean'}).sort_values('checkout_price', ascending = False)

final_promo = email_no_promo.merge(email_promo, on = 'category', how = 'left', suffixes = ('_no_promo','_promo'))

# mean price variation

final_promo['price_vs_promo'] = ((email_promo['checkout_price'] / email_no_promo['checkout_price']) -1)*100

clean = final_promo.fillna('No email promotion')
print(clean)

# sum up: number of orders by center_id and category/cuisine

orders_region_cui = df_meal_ffc[['region_code','cuisine','num_orders']].groupby(['region_code','cuisine'], as_index = False).agg({'num_orders':'sum'}).sort_values(['region_code','num_orders'], ascending = [True, False])
orders_region_cui['region_code'] = orders_region_cui['region_code'].astype(str)
sns.barplot(x ='cuisine', y ='num_orders',hue = 'region_code', data = orders_region_cui,
            palette ='plasma').set_title('Number of orders by Cuisine by Region')
plt.xlabel("Cuisine", fontsize=12)
plt.xticks(rotation=45)
plt.ylabel("Number of orders", fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

orders_region_cat = df_meal_ffc[['region_code','category','num_orders']].groupby(['region_code','category'], as_index = False).agg({'num_orders':'sum'}).sort_values(['region_code','num_orders'], ascending = [True, False])
print(orders_region_cat.head())
