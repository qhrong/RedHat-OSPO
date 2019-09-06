# This python file is used as a function to transform any subset of the OSPO data.

# Import modules
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from datetime import datetime

# Define new dataframe to store transformed df
new_df = pd.DataFrame()

def data_transformer(df):

    # Any df applied in this function should only the following columns:
    '''
    'repos_id0', 'committer_name', 'committer_email', 'committer_date',
       'added', 'removed', 'whitespace', 'filename', 'date_attempted', 'git',
       'repos_name', 'committer_affiliation', 'committer_year',
       'committer_month', 'committer_day'
    '''

    # Extract the email name
    df["email_name"] = df['committer_email'].str.split('@', 1, expand=True)[0]

    # Transfer string to date format
    res=[]
    for ele in df["committer_date"]:
        try:
            res.append(pd.to_datetime(str(ele),"%Y-%m-%d %H:%M:%S%Z"))
        except:
            res.append(pd.to_datetime(str(ele),infer_datetime_format=True))
    df['committer_date']=res

    # Create columns for year, month, day
    year=[]
    month=[]
    day=[]
    for ele in df["committer_date"]:
        year.append(ele.year)
        month.append(ele.month)
        day.append(ele.day)
    df["committer_year"]=year
    df["committer_month"]=month
    df["committer_day"]=day

    # Reclassify people's affiliation
    df['committer_affiliation'] = df['committer_email'].str.split('@', 1, expand=True)[1]
    df['committer_affiliation'] = df['committer_affiliation'].str.split('[.]', 1, expand=True)[0]
    df['committer_affiliation'] = np.where(df['committer_affiliation'] == 'redhat', 'redhat', 'RH/V')

    # 1. Percentage of commit made on each day / each part of day ------------------------------------------------------
    # Feature of weekday
    day = []
    for ele in df['committer_date']:  # weekday=0, weekend=1
        day.append(ele.weekday())
    df['weekday'] = day

    # Feature of morning or evening
    def get_part_of_day(hour):
        if 5 <= hour <= 11:
            return "morning"
        elif 12 <= hour <= 17:
            return "afternoon"
        elif 18 <= hour <= 22:
            return "evening"
        else:
            return "night"

    res = []
    for ele in df['committer_date']:
        res.append(get_part_of_day(ele.hour))
    df['day_part'] = res

    # A df counting how many commits from weekday/weekend for each committer
    weekday_df = df.groupby(['committer_name', 'weekday'])['committer_name'].count().reset_index(name='count')
    # A df counting how many commits from each part of day for each committer
    daypart_df = df.groupby(['committer_name', 'day_part'])['committer_name'].count().reset_index(name='count')
    # Calculate what percentage a committer commit in weekday/weekend
    committer_weekday = weekday_df.groupby(['committer_name', 'weekday']).agg({'count': 'sum'})
    committer_weekday = committer_weekday.groupby(level=0).apply(lambda x: x / float(x.sum()))
    # Calculate what percentage a committer commit in each part of day
    committer_daypart = daypart_df.groupby(['committer_name', 'day_part']).agg({'count': 'sum'})
    committer_daypart = committer_daypart.groupby(level=0).apply(lambda x: x / float(x.sum()))

    # Reframing the weekday dataframe
    committer_weekday = committer_weekday.pivot_table(columns='weekday', index='committer_name', values='count',
                                                      fill_value=0)
    # Reframing the daypart dataframe
    committer_daypart = committer_daypart.pivot_table(columns='day_part', index='committer_name', values='count',
                                                      fill_value=0)

    new_df = pd.concat([committer_weekday, committer_daypart], axis=1)

    # 2. Committer's contribution features -----------------------------------------------------------------------------
    # The creation of features are on project level. If people have participated in multiple projects, they will be added up, for % of changes.

    # 2.1 Yearly Rank and project count
    def project_year(ele):
        subdf = df[df['repos_name'] == ele]
        year_df = subdf.groupby(['committer_year', 'committer_name'])['committer_affiliation'].count().reset_index(
            name='count')
        # Rank column
        year_df['rank_yr'] = year_df.groupby(['committer_year'])['count'].rank(ascending=False).astype(int)
        year_df = year_df.groupby('committer_name')['rank_yr'].sum()
        return (year_df)

    res = project_year('camel')  # use the first one to set storage format
    project_name = list(df['repos_name'].unique())

    for ele in project_name[1:]:  # delete camel because already used as storage format
        res = res.append(project_year(ele))

    # Edit format
    res['name'] = res.index
    res = pd.DataFrame(res)

    # Summarize the rank and append to new_df
    count = res.groupby('committer_name').size() #how many project participated 
    res = res.groupby('committer_name')['rank_yr'].sum()
    res = pd.DataFrame(res)
    res['count'] = count

    new_df = pd.concat([new_df, res], axis=1)


     # 2.2 Sum of percentage of changes in each project

    # for ele in df['repos_name'].unique():
    def project_df(ele):  # ele is the project name
        # Data frame for this project
        df_proj = df[df['repos_name'] == ele]
        df_proj2 = df_proj.groupby('committer_name')['added', 'removed', 'whitespace'].sum()
        df_proj2 = pd.merge(df_proj2, df_proj.groupby('committer_name').size().reset_index(name='touched'),
                            on='committer_name')

        # Transfer number of changes to percentage / total number in this project
        df_proj2['added'] = df_proj2['added'] / sum(df_proj2['added'])
        df_proj2['removed'] = df_proj2['removed'] / sum(df_proj2['removed'])
        df_proj2['whitespace'] = df_proj2['whitespace'] / sum(df_proj2['whitespace'])
        df_proj2['touched'] = df_proj2['touched'] / sum(df_proj2['touched'])
        return (df_proj2)

    res = project_df('camel')  # use the first one to set storage format
    for ele in project_name[1:]:  # delete camel because already used as storage format
        res = res.append(project_df(ele))

    res = res.groupby('committer_name')['added', 'removed', 'whitespace', 'touched'].sum()

    new_df = pd.concat([new_df, res], axis=1)
    # NaN means 0 of changes so no percentage


     # 3. Nearest neighbor ----------------------------------------------------------------------------------------------

    def knn_committer(df, ele):  # df: original; ele: project name

        # filter out repos with count>2
        if len(df[df['repos_name'] == ele].groupby('committer_name')) > 2:

            knn_df = df[df['repos_name'] == ele]
            knn_df = knn_df.groupby('committer_name')['added', 'removed', 'whitespace'].sum()

            # normalized the dataframe
            for i in ['added', 'removed', 'whitespace']:
                knn_df[i] = (knn_df[i] - min(knn_df[i]))

            knn_df.fillna(0)

            # find the nearest neighbor
            nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(knn_df)
            distances, indices = nbrs.kneighbors(knn_df)
            res = []
            for i in range(len(knn_df['added'])):
                res.append(indices[i][1])
            knn_df['neighbor'] = res

            knn_df['committer_name'] = knn_df.index

            # Append neighbors' names as a new column
            res = []
            names = []
            # some of the first items in indices are the committer themselves, some are not. Need a if-else to filter out
            for i in range(len(indices)):
                if indices[i][0] == i:
                    res.append(indices[i][1])
                else:
                    res.append(indices[i][0])  # save the neighbor other than itself
                names.append(knn_df['committer_name'][res[i]])
            knn_df['neighbor_name'] = names

        else:

            knn_df = df[df['repos_name'] == ele]
            knn_df = knn_df.groupby('committer_name')['added', 'removed', 'whitespace'].sum()

            # standardize the dataframe
            for i in ['added', 'removed', 'whitespace']:
                knn_df[i] = (knn_df[i] - np.mean(knn_df[i]))

            knn_df.fillna(0)

            knn_df['neighbor'] = ['Na'] * len(knn_df)
            knn_df['committer_name'] = knn_df.index
            knn_df['neighbor_name'] = ['Na'] * len(knn_df)

        return (knn_df)

    # Iterate for each project
    ans = knn_committer(df, 'camel')

    for ele in project_name[1:]:
        tmp = knn_committer(df.fillna(0), ele)
        ans = ans.append(tmp)

    # Add a column to indicate the neighbor's affiliation
    res = []

    for ele in ans['neighbor_name']:
        try:
            compare_tmp = df['committer_affiliation'][df['committer_name'] == ele]
            res.append(compare_tmp.iloc[0])
        except:
            res.append('Na')
    ans['affiliation'] = res

    # Numerize the affiliation column for group by
    ans['affiliation'] = np.where(ans['affiliation'] == 'RH/V', 0, 1)

    new_df['affiliation'] = ans.groupby('committer_name')['affiliation'].sum()
    # The affiliation column indicates how many neighbor of this committer is from Red Hat.

    # 4. Encodings for repos name --------------------------------------------------------------------------------------

    encoding_tmp = pd.DataFrame(df.groupby(['committer_name', 'repos_name']).count()['repos_id'])

    encoding_tmp = encoding_tmp.pivot_table(columns='repos_name', index='committer_name', values='repos_id',
                                            fill_value=0)

    new_df = pd.concat([new_df, encoding_tmp], axis=1)

    # Save as a csv
    new_df.to_csv('transformed_sample.csv')

    # Output of the function -------------------------------------------------------------------------------------------
    return(new_df)


