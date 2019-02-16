import pandas as pd

"""
BASED on MovieLens data sets:
F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets:
History and Context. ACM Transactions on Interactive Intelligent
Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages.
DOI=http://dx.doi.org/10.1145/2827872

https://grouplens.org/datasets/movielens/100k/
"""


#IMPORTING DATABASE
# u.date -- > csv with user id and their ratings on item_id
data = pd.read_csv('u.data', sep='\t', names=['userID','item_id','rating','time stmp'])

#u.item -- > film details, here only title and id needed
df = pd.read_csv('u.item', sep= '|',encoding = "ISO-8859-1",names=[n for n in range(24)])
movie_title = pd.DataFrame()
movie_title['item_id'] = df.iloc[:,0]
movie_title['film_title'] = df.iloc[:,1]

#merging for better readibilty
data = pd.merge(data, movie_title, on='item_id')
data.drop('time stmp', axis=1,inplace=True)

#ratings - mean rate and number of votes
ratings = data.loc[:,['film_title','rating']]
ratings =  ratings.groupby('film_title').mean()
ratings['number of votes'] = data['film_title'].value_counts()

#table for comparisons
ratings_matrix = data.pivot_table(index='userID', columns='film_title', values='rating')

#function for finding correlation - PEOPLE WHO LIKED THIS MOVE ALSO LIKED
def also_liked(movie):
    corrmovie = pd.DataFrame(ratings_matrix.corrwith(ratings_matrix[movie]), columns=['correlation'])
    corrmovie.dropna(inplace=True)
    corrmovie = corrmovie.join(ratings['number of votes'])
    corrmovie = corrmovie[corrmovie['number of votes']>100].sort_values(['correlation'], ascending=False)
    return corrmovie[1:].head(7)

#example - correlation for one movie
movie  =  '12 Angry Men (1957)'
print(also_liked(movie))


