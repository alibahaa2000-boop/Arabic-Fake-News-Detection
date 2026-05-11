from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from bs4 import BeautifulSoup
import pandas as pd
import os
from datetime import datetime
import time
import csv
import hashlib

# ================= CONFIG =================
SITES = {
    "aljazeera": {"url": "https://www.aljazeera.net", "path": "/news/"},
    "bbc_arabic": {"url": "https://www.bbc.com/arabic", "path": "/arabic/articles/"},
    "cnn_arabic": {"url": "https://arabic.cnn.com", "path": None},
    "rt_arabic": {"url": "https://arabic.rt.com/", "path": None},
    "dw_arabic": {"url": "https://www.dw.com/ar/", "path": None},
    "reuters_ar": {"url": "https://arabic.reuters.com/", "path": None},
    "alquds": {"url": "https://www.alquds.com/", "path": None},
    "aawsat": {"url": "https://aawsat.com/", "path": None},
    "alhurra": {"url": "https://www.alhurra.com/", "path": None},
    "trtarabi": {"url": "https://www.trtarabi.com/", "path": None},
    "cgtn_arabic": {"url": "https://arabic.cgtn.com/", "path": None},
    "sputnik_arabic": {"url": "https://sputnikarabic.ae/", "path": None},
    "asharq_news": {"url": "https://asharq.com/", "path": None},
    "alaraby": {"url": "https://www.alaraby.com/", "path": None},
    "aa_ar": {"url": "https://www.aa.com.tr/ar", "path": None},
    "euronews_arabic": {"url": "https://arabic.euronews.com/", "path": None},
    "syria_tv": {"url": "https://www.syria.tv/", "path": None},
    "aliraqiya": {"url": "https://imn.iq/", "path": None},
    "alekhbariya": {"url": "https://www.alekhbariya.net/ar", "path": None},
    "almamlaka": {"url": "https://www.almamlakatv.com/", "path": None},
    "kuna": {"url": "https://www.kuna.net.kw/", "path": None},
    "oman_news": {"url": "https://omannews.gov.om/", "path": None},
    "yementv": {"url": "https://yementv.tv/", "path": None},
    "wafa": {"url": "https://www.wafa.ps/", "path": None},
    "al24": {"url": "https://al24news.dz/", "path": None},
    "libya_tv": {"url": "https://alrasmia.ly/", "path": None},
    "sudan_tv": {"url": "https://sudan-tv.net/", "path": None},
    "elmourabitoun": {"url": "https://elmourabiton.tv/ar/", "path": None},
    "skynews_arabia": {"url": "https://www.skynewsarabia.com/", "path": None},
}

# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, "selenium_dataset")
os.makedirs(DATA_FOLDER, exist_ok=True)
DATASET_FILE = os.path.join(DATA_FOLDER, "selenium_dataset.csv")

# ================= DRIVER =================
def create_driver():
    options = Options()
    options.add_argument("--headless=new")  # أفضل مع كروم الجديد
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1366,768")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    return webdriver.Chrome(options=options)

driver = create_driver()

# ================= LOAD OLD DATA (FIXED + LINK) =================
EXPECTED_COLS = ["site", "title", "link", "scraped_at", "content"]

def load_old_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=EXPECTED_COLS)

    df = None
    # جرّب ; وبعدين , لأن مرات الملف ينحفظ بفاصل مختلف
    for sep in [";", ","]:
        try:
            tmp = pd.read_csv(path, sep=sep, encoding="utf-8-sig")
            # إذا طلع عمود واحد بس غالباً الفاصل غلط
            if tmp.shape[1] == 1 and sep == ";":
                continue
            df = tmp
            break
        except Exception:
            continue

    if df is None:
        print("⚠️ لم أستطع قراءة ملف الداتا القديم، سيتم إنشاء ملف جديد.")
        return pd.DataFrame(columns=EXPECTED_COLS)

    # توحيد أسماء الأعمدة
    df.columns = [str(c).strip().lower() for c in df.columns]

    # إذا الملف قديم وما يحتوي link، أضفه حتى ما ينهار الكود
    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = ""

    # رتّب الأعمدة
    df = df[EXPECTED_COLS]
    return df

old_df = load_old_dataset(DATASET_FILE)

# ✅ منع التكرار: روابط + هاش محتوى
old_links = set(old_df["link"].dropna().astype(str))

old_content_hashes = set()
for c in old_df["content"].dropna().astype(str):
    h = hashlib.sha256(c[:1000].encode("utf-8", errors="ignore")).hexdigest()
    old_content_hashes.add(h)

new_content_hashes = set()
new_rows = []

# ================= FUNCTIONS =================
def scroll_page(driver, scrolls=6):
    for _ in range(scrolls):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1.2)

def extract_article_text(url, site):
    try:
        driver.get(url)
        WebDriverWait(driver, 6).until(
            EC.presence_of_element_located((By.TAG_NAME, "p"))
        )
    except Exception:
        return ""

    soup = BeautifulSoup(driver.page_source, "html.parser")

    selectors = {
        "cnn_arabic": ("div", "article__content"),
        "skynews_arabia": ("div", "article-body"),
        "rt_arabic": ("div", "field-item"),
        "dw_arabic": ("div", "longText"),
        "reuters_ar": ("div", "ArticleBody__content___2gQno"),
    }

    if site in selectors:
        tag, cls = selectors[site]
        article = soup.find(tag, class_=cls)
    else:
        article = soup

    paragraphs = article.find_all("p") if article else soup.find_all("p")
    text = " ".join(p.get_text(strip=True) for p in paragraphs)
    return text.strip()

# ================= SCRAPING =================
for site, cfg in SITES.items():
    print(f" Scraping: {site}")
    try:
        driver.get(cfg["url"])
        scroll_page(driver)
        soup = BeautifulSoup(driver.page_source, "html.parser")

        for a in soup.find_all("a", href=True):
            title = a.get_text(strip=True)
            link = a["href"]

            if not title or not link:
                continue

            # تحويل الروابط النسبية إلى مطلقة
            if not link.startswith("http"):
                link = cfg["url"].rstrip("/") + "/" + link.lstrip("/")

            # فلترة مسارات خاصة ببعض المواقع
            if site in ["aljazeera", "bbc_arabic"] and cfg["path"]:
                if cfg["path"] not in link:
                    continue

            # فلترة روابط غير مرغوبة
            if any(x in link.lower() for x in ["rss", "telegram", "video", "live"]):
                continue

            # ✅ منع التكرار بالرابط
            if link in old_links:
                continue

            content = extract_article_text(link, site)

            if not content or len(content.split()) < 50:
                continue

            # ✅ منع التكرار بالمحتوى أيضاً (احتياط)
            content_hash = hashlib.sha256(
                content[:1000].encode("utf-8", errors="ignore")
            ).hexdigest()

            if content_hash in old_content_hashes or content_hash in new_content_hashes:
                continue

            new_content_hashes.add(content_hash)
            old_links.add(link)  # مهم حتى لا يتكرر داخل نفس التشغيل

            new_rows.append({
                "site": site,
                "title": title,
                "link": link,
                "scraped_at": datetime.now().isoformat(),
                "content": content
            })

    except Exception as e:
        print(f" Error scraping {site}: {e}")

# ================= SAVE =================
driver.quit()

final_df = pd.concat([old_df, pd.DataFrame(new_rows)], ignore_index=True)
final_df.to_csv(
    DATASET_FILE,
    index=False,
    encoding="utf-8-sig",
    sep=";",
    quoting=csv.QUOTE_MINIMAL,
    escapechar="\\"
)

print(" تم الحفظ بنجاح")
print(" عدد الصفوف:", len(final_df))
print(" عدد الصفوف الجديدة:", len(new_rows))
