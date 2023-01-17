import requests
req = requests.get("https://query.data.world/s/gwuvcaposio5kejfu322u3znsrz637")
url_content = req.content

with open('static/dataset/atlantic.csv', 'wb') as csv_file:
    csv_file.write(url_content)
    
    
    
    
    
df = pd.read_csv('atlantic (2).csv')
df.drop(['status_of_system', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8','Unnamed: 9', 
        'Unnamed: 10', 'Unnamed: 11','Unnamed: 12', 'Unnamed: 13', 'Unnamed: 14', 
        'Unnamed: 15', 'Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18'], inplace=True, axis=1)
df['date'] = pd.to_datetime(df['date'], format = "%Y%m%d").dt.strftime('%Y-%m-%d')
df['tmrw windspeed'] =  df['max_sustained_wind'].shift(-1)
df.to_csv('cleanedHurricaneData.csv', index=False)
df['date'] = df['date'].apply(lambda x: float(x.split()[0].replace('-', '')))
df['latitude'] = df['latitude'].map(lambda x: float(x.rstrip('NEWS')))
df['longitude'] = df['longitude'].map(lambda x: float(x.rstrip('NEWS')))
df = df.where(df > 0, 0)
df['max_sustained_wind'] =  df['max_sustained_wind'].fillna(method='ffill', limit=1000)
df['max_sustained_wind'] =  df['max_sustained_wind'].fillna(method='bfill', limit=1000)
df['central_pressure'] = df['central_pressure'].fillna(method='ffill', limit=1000)
df['central_pressure'] = df['central_pressure'].fillna(method='bfill', limit=1000)
df['tmrw windspeed'] = df['tmrw windspeed'].fillna(method='ffill', limit=1000)
df['tmrw windspeed'] = df['tmrw windspeed'].fillna(method='bfill', limit=1000)

print(df)
