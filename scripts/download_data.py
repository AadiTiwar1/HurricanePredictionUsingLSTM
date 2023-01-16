import requests
req = requests.get("https://query.data.world/s/gwuvcaposio5kejfu322u3znsrz637")
url_content = req.content

with open('static/dataset/atlantic.csv', 'wb') as csv_file:
    csv_file.write(url_content)
