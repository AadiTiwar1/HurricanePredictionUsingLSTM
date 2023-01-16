import requests
req = requests.get("https://query.data.world/s/gwuvcaposio5kejfu322u3znsrz637")
url_content = req.content
csv_file = open('atlantic (2).csv', 'wb')
csv_file.write(url_content)
csv_file.close()
