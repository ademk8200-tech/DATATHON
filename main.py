import pandas as pd
from ydata_profiling import ProfileReport

# Örnek olarak klasik Titanic veri setini çekiyoruz
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Profiling raporunu oluştur (hızlı olması için minimal ayarlar da eklenebilir)
profile = ProfileReport(df, title="Datathon Öncesi Pratik Raporu")

# HTML olarak kaydet
profile.to_file("veri_raporu.html")